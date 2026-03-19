import os
import pickle
import traceback
from groq import Groq
from sentence_transformers import SentenceTransformer, util

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import get_categories, route_question
from immutable_facts import load_facts, check_immutable

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_KEY")
HIERARCHY_FOLDER = "nli_hierarchy_v7"
TOP_K_NODES = 7

groq_client = Groq(api_key=GROQ_API_KEY)

print("Loading sentence embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready.\n")
print("Loading immutable facts...")
load_facts(embedder)
print("Done.\n")


def get_groq_answer(question: str, context_rules: list) -> str:
    context_text = "\n".join(f"- {r}" for r in context_rules)
    prompt = (
        "You are a knowledgeable assistant specializing in Indian Central Government service rules and policies.\n\n"
        "Rules from official DoPT documents:\n" + context_text +
        "\n\nAnswer this question clearly and concisely. Use 1-2 sentences for simple factual questions, "
        "and 3-5 sentences for complex questions that require explaining conditions or exceptions. "
        "Do not repeat information. Start with the direct answer. Do not mention OM numbers.\n\n"
        "Question: " + question
    )
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def get_groq_answer_no_context(question: str) -> str:
    prompt = (
        "You are a helpful assistant for anyone with queries about Indian Central Government service rules and policies. "
        "If the message is a greeting, respond with just: Hi! How can I help you? "
        "Otherwise answer clearly and concisely. Use 1-2 sentences for simple factual questions, "
        "3-5 sentences for complex ones. Do not repeat information.\n\n"
        "Question: " + question
    )
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


_hierarchy_cache = {}

def load_hierarchy(category: str):
    if category in _hierarchy_cache:
        return _hierarchy_cache[category]
    for ext in [".pkl", ".pickle", ""]:
        path = os.path.join(HIERARCHY_FOLDER, f"{category}{ext}")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    G = pickle.load(f)
                _hierarchy_cache[category] = G
                return G
            except Exception:
                pass
    _hierarchy_cache[category] = None
    return None


_node_embedding_cache = {}

def retrieve_nodes(G, question: str, top_k: int = TOP_K_NODES) -> list:
    if G is None:
        return []
    nodes_info = [
        (nid, G.nodes[nid]['text'], G.nodes[nid].get('level', 1))
        for nid in G.nodes
        if nid != "ROOT" and G.nodes[nid].get('text')
    ]
    if not nodes_info:
        return []
    node_ids, node_texts, node_levels = zip(*nodes_info)
    graph_id = id(G)
    if graph_id not in _node_embedding_cache:
        _node_embedding_cache[graph_id] = embedder.encode(
            list(node_texts), convert_to_tensor=True, show_progress_bar=False
        )
    node_embeddings = _node_embedding_cache[graph_id]
    q_emb = embedder.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, node_embeddings)[0].tolist()
    adjusted = [
        (s + (0.08 if lvl == 1 else 0.0), txt)
        for s, txt, lvl in zip(scores, node_texts, node_levels)
    ]
    adjusted.sort(reverse=True)
    top_scores = [s for s, _ in adjusted[:top_k]]
    if top_scores and (sum(top_scores) / len(top_scores)) < 0.25:
        return []
    seen = set()
    top_rules = []
    for _, text in adjusted:
        key = text[:80]
        if key not in seen:
            seen.add(key)
            top_rules.append(text)
        if len(top_rules) >= top_k:
            break
    return top_rules


def main():
    categories = get_categories()
    print("DoPT Policy Assistant  (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        immutable_answer = check_immutable(user_input, embedder)
        if immutable_answer:
            print(f"\nAssistant: {immutable_answer}\n")
            continue

        category = route_question(user_input, categories)

        if category == "__clarify__":
            answer = get_groq_answer_no_context(user_input)
            print(f"\nAssistant: {answer}\n")
            continue

        try:
            G = load_hierarchy(category)
            if G and len(G.nodes) > 2:
                rules = retrieve_nodes(G, user_input)
                if rules:
                    answer = get_groq_answer(user_input, rules)
                else:
                    answer = get_groq_answer_no_context(user_input)
            else:
                answer = get_groq_answer_no_context(user_input)
        except Exception:
            traceback.print_exc()
            answer = get_groq_answer_no_context(user_input)

        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()