import os
import re
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
print("Loading NLI model...")
_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
_nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
_nli_model.eval()
print("NLI model loaded.\n")

IMMUTABLE_HYPOTHESIS = "This is a fixed rule that applies universally with no exceptions or conditions."

ENTAILMENT_THRESHOLD = 0.75

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

def is_garbage(line):
    if not line.strip():
        return True
    total = len(line)
    alpha = sum(1 for c in line if c.isalpha() or c.isspace())
    if total > 0 and (alpha / total) < 0.62:
        return True
    return False

def get_sentences(text):
    """Clean and split text into sentences."""
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
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    clean = []
    for s in sentences:
        s = s.strip()
        if len(s) < 40 or len(s) > 400:
            continue
        if any(re.search(p, s, re.IGNORECASE) for p in SKIP_PATTERNS):
            continue
        if is_garbage(s):
            continue
        clean.append(s)
    return clean


def is_immutable(sentence):
    inputs = _tokenizer(
        sentence,
        IMMUTABLE_HYPOTHESIS,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        logits = _nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    entailment_score = probs[2].item()
    return entailment_score >= ENTAILMENT_THRESHOLD, entailment_score


def extract_immutable_facts(text, category, max_facts=30):
    sentences = get_sentences(text)
    print(f"  {len(sentences)} candidate sentences to check...")

    facts = []
    seen = set()
    for i, sent in enumerate(sentences):
        key = sent[:80].lower()
        if key in seen:
            continue
        seen.add(key)

        immutable, score = is_immutable(sent)
        if immutable:
            facts.append((sent, round(score, 3)))
            print(f"    [IMMUTABLE {score:.2f}] {sent[:90]}...")

        if len(facts) >= max_facts:
            break

    return facts

TEXT_FOLDER = "extracted_text_final2"
SAVE_PATH = "immutable_facts.pkl"

all_facts = {}  

text_files = sorted(f for f in os.listdir(TEXT_FOLDER) if f.endswith(".txt"))

for filename in text_files:
    category = filename.replace(".txt", "").strip()
    input_path = os.path.join(TEXT_FOLDER, filename)

    print(f"\n{'='*50}")
    print(f"Category: {category}")
    print(f"{'='*50}")

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    facts = extract_immutable_facts(text, category)
    all_facts[category] = facts
    print(f"  Found {len(facts)} immutable facts")

with open(SAVE_PATH, "wb") as f:
    pickle.dump(all_facts, f)

print(f"\nDone! Immutable facts saved to {SAVE_PATH}")
print(f"Total facts across all categories: {sum(len(v) for v in all_facts.values())}")