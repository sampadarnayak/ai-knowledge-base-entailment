import os
import re
import time
import pickle

import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

print("Loading NLI model...")
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
_nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
_nli_model.eval()
print("NLI model loaded.")

print("Loading sentence embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedder loaded.\n")


def nli_pair(premise: str, hypothesis: str):
    
    inputs = _tokenizer(
        premise, hypothesis,
        return_tensors="pt", truncation=True, max_length=512
    )
    with torch.no_grad():
        logits = _nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
    idx = probs.argmax().item()
    return label_map[idx], probs[idx].item()


CATEGORY_TEMPLATES = {
    "retirement": "Retirement conditions for government employees vary by service type, qualifying service, and applicable rules",
    "promotion": "Promotion eligibility and benchmarks in government service depend on APAR grading, years of service, DPC recommendation, and vacancy",
    "deputation": "Deputation period and allowable tenure vary by post level, organization type, cadre, and government approval",
    "disciplinary": "Disciplinary action and vigilance clearance conditions vary by nature of charges, penalty imposed, and competent authority decision",
    "incentives": "Incentives and special allowances for government employees vary by service type, qualification, and applicable government orders",
    "pay matters": "Pay fixation and recovery rules vary by service level, nature of payment, and applicable pay commission orders",
    "recruitment": "Recruitment norms and eligibility conditions vary by post level, service type, and applicable recruitment rules",
    "reservation": "Reservation provisions and roster rules vary by category, type of recruitment, and applicable government orders",
    "training": "Training requirements and probation conditions vary by service level, department, and applicable probation rules",
    "appointment": "Appointment conditions and committee composition vary by post level, service type, and applicable government guidelines",
    "cadre review": "Cadre strength and review norms for IAS vary by state cadre, post level, and applicable cadre rules",
}

def make_parent_text(fact1: str, fact2: str, category: str) -> str:
    
    if category in CATEGORY_TEMPLATES:
        return CATEGORY_TEMPLATES[category]

    words = fact1.split()
    subject = " ".join(words[:min(5, len(words))])
    return f"Rules regarding '{subject}...' have conditional variations depending on service type and applicable government orders"


SKIP_PATTERNS = [
    r'^No\.\s*[\d/]+', r'^F\.No', r'^Dated', r'^Government of India',
    r'^Ministry of', r'^Department of', r'^Subject\s*:',
    r'^The undersigned', r'^OFFICE MEMORANDUM',
    r'^\s*(To|From|Copy to|Sir|Madam|Yours faithfully)\s*$',
    r'^\s*\d+\s*$', r'^(Table|Annexure|Appendix|Schedule|Copy forwarded)',
    r'^(Under Secretary|Deputy Secretary|Joint Secretary|Director)\b',
    r'^\(.*\)\s*$', r'^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}',
    r'^\s*[-=*_]{3,}\s*$',
]

RULE_KEYWORDS = [
    "shall", "will not", "may not", "must", "cannot", "should",
    "eligible", "entitled", "permitted", "allowed", "prohibited",
    "period of", "years of service", "months", "days",
    "age of", "maximum", "minimum", "subject to", "provided that",
    "deputation", "promotion", "retirement", "leave", "pay",
    "service", "cadre", "post", "grade", "appointment",
    "DPC", "APAR", "benchmark", "penalty", "dismissal",
    "roster", "reservation", "SC", "ST", "OBC",
    "increment", "allowance", "salary", "qualifying",
    "probation", "training", "vigilance", "clearance",
]

def is_garbage(line: str) -> bool:
    if not line.strip():
        return True
    total = len(line)
    alpha = sum(1 for c in line if c.isalpha() or c.isspace())
    if total > 0 and (alpha / total) < 0.62:
        return True
    return False

def extract_rules(text: str, max_rules: int = 120) -> list:
    
    lines = text.split('\n')
    joined = []
    buffer = ""
    for line in lines:
        line = line.strip()
        if not line or is_garbage(line):
            if buffer:
                joined.append(buffer)
                buffer = ""
            continue
        if buffer:
            if not re.search(r'[.!?:]\s*$', buffer):
                buffer = buffer + " " + line
            else:
                joined.append(buffer)
                buffer = line
        else:
            buffer = line
    if buffer:
        joined.append(buffer)

    text = '\n'.join(joined)

    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    rules = []
    seen = set()
    for sent in raw_sentences:
        sent = sent.strip()
        if len(sent) < 50 or len(sent) > 600:
            continue
        if any(re.search(p, sent, re.IGNORECASE) for p in SKIP_PATTERNS):
            continue
        if is_garbage(sent):
            continue
        lower = sent.lower()
        if not any(kw in lower for kw in RULE_KEYWORDS):
            continue
        key = re.sub(r'\s+', ' ', lower)[:120]
        if key in seen:
            continue
        seen.add(key)
        rules.append(sent)
        if len(rules) >= max_rules:
            break

    return rules


def build_hierarchy(rules: list, category: str) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node("ROOT", text="All Government Service Rules", level=0)

    if not rules:
        return G

    print(f"  Computing embeddings for {len(rules)} rules...")
    embeddings = embedder.encode(rules, convert_to_tensor=True, show_progress_bar=False)

    existing_parent_texts = set()

    for i, new_fact in enumerate(rules):
        if i % 20 == 0 and i > 0:
            print(f"    {i}/{len(rules)} rules processed...")

        if new_fact in G.nodes:
            continue

        non_root = [n for n in G.nodes if n != "ROOT" and n in rules]
        placed = False

        if non_root:
            indices = [rules.index(n) for n in non_root if n in rules]
            if indices:
                cand_embs = embeddings[indices]
                new_emb = embeddings[i].unsqueeze(0)
                sims = util.cos_sim(new_emb, cand_embs)[0]

                top_k = min(5, len(indices))
                top_idx = sims.topk(top_k).indices.tolist()

                for rank_idx in top_idx:
                    sim_score = sims[rank_idx].item()

                    if sim_score < 0.30:
                        continue

                    existing_node = non_root[rank_idx] if rank_idx < len(non_root) else None
                    if not existing_node or existing_node not in G.nodes:
                        continue

                    label, score = nli_pair(G.nodes[existing_node]['text'], new_fact)

                    if label == "entailment" and score > 0.52:
                        new_level = G.nodes[existing_node]['level'] + 1
                        G.add_node(new_fact, text=new_fact, level=new_level)
                        G.add_edge(existing_node, new_fact, relation="specializes")
                        placed = True
                        break

                    elif label == "contradiction" and score > 0.52:
                        parent_text = make_parent_text(
                            G.nodes[existing_node]['text'], new_fact, category
                        )

                        
                        if parent_text in existing_parent_texts:
                            
                            for pnode in G.nodes:
                                if G.nodes[pnode].get('text') == parent_text:
                                    existing_level = G.nodes[pnode]['level']
                                    G.add_node(new_fact, text=new_fact, level=existing_level + 1)
                                    G.add_edge(pnode, new_fact, relation="specializes")
                                    placed = True
                                    break
                        else:
                            existing_parent_texts.add(parent_text)
                            existing_level = G.nodes[existing_node]['level']
                            parent_id = f"parent_{len(G.nodes)}"
                            G.add_node(parent_id, text=parent_text, level=existing_level)
                            G.add_edge("ROOT", parent_id, relation="entails")
                            if G.has_edge("ROOT", existing_node):
                                G.remove_edge("ROOT", existing_node)
                            G.add_edge(parent_id, existing_node, relation="specializes")
                            G.add_node(new_fact, text=new_fact, level=existing_level + 1)
                            G.add_edge(parent_id, new_fact, relation="specializes")
                            placed = True

                        if placed:
                            break

        if not placed:
            G.add_node(new_fact, text=new_fact, level=1)
            G.add_edge("ROOT", new_fact, relation="entails")

    return G


TEXT_FOLDER = "extracted_text_final2"
SAVE_FOLDER = "nli_hierarchy_v7"
os.makedirs(SAVE_FOLDER, exist_ok=True)

text_files = sorted(f for f in os.listdir(TEXT_FOLDER) if f.endswith(".txt"))

for filename in text_files:
    start = time.time()
    category = filename.replace(".txt", "").strip()
    input_path = os.path.join(TEXT_FOLDER, filename)
    save_path = os.path.join(SAVE_FOLDER, f"{category}.pkl")

    print(f"\n{'='*50}")
    print(f"Category: {category}")
    print(f"{'='*50}")

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    rules = extract_rules(text, max_rules=120)
    print(f"Extracted {len(rules)} rules")

    if not rules:
        print(f"  WARNING: No rules — skipping.")
        continue

    print("  Samples:")
    for r in rules[:2]:
        print(f"    → {r[:110]}")

    G = build_hierarchy(rules, category)

    with open(save_path, "wb") as f:
        pickle.dump(G, f)

    elapsed = time.time() - start
    level1 = sum(1 for n in G.nodes if n != "ROOT" and G.nodes[n].get('level') == 1)
    level2 = sum(1 for n in G.nodes if G.nodes[n].get('level') == 2)
    parents = sum(1 for n in G.nodes if n.startswith("parent_"))

    print(f"Saved: {save_path}")
    print(f"  Nodes: {len(G.nodes)} | Level-1: {level1} | Level-2: {level2} | Parent nodes: {parents} | Time: {elapsed:.1f}s")

print("\n✓ Done! Hierarchies saved to nli_hierarchy_v7/")
