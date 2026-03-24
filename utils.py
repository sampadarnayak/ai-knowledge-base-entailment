import os
import pickle
from sentence_transformers import SentenceTransformer, util
import torch

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

_category_embeddings = {}
_category_name_embeddings = {}

def build_category_embeddings(hierarchy_folder):
    global _category_embeddings, _category_name_embeddings
    embedder = get_embedder()
    _category_embeddings = {}
    _category_name_embeddings = {}

    categories = []
    for fname in os.listdir(hierarchy_folder):
        if not (fname.endswith(".pkl") or fname.endswith(".pickle")):
            continue
        category = fname.replace(".pkl", "").replace(".pickle", "").strip()
        path = os.path.join(hierarchy_folder, fname)
        try:
            with open(path, "rb") as f:
                G = pickle.load(f)
            texts = [
                G.nodes[n]["text"] for n in G.nodes
                if n != "ROOT" and G.nodes[n].get("text")
            ]
            if texts:
                embs = embedder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
                _category_embeddings[category] = embs.mean(dim=0)
                categories.append(category)
        except Exception as e:
            print(f"Warning: could not load {fname}: {e}")

    if categories:
        name_embs = embedder.encode(categories, convert_to_tensor=True, show_progress_bar=False)
        for i, cat in enumerate(categories):
            _category_name_embeddings[cat] = name_embs[i]

    return categories


def get_categories():
    return list(_category_embeddings.keys())


def route_question(question: str, categories: list) -> str:
    if not _category_embeddings:
        return "__clarify__"

    embedder = get_embedder()
    q_emb = embedder.encode(question, convert_to_tensor=True)

    best_category = None
    best_score = -1.0

    for category in _category_embeddings:
       
        content_score = util.cos_sim(q_emb, _category_embeddings[category]).item()
        
        name_score = util.cos_sim(q_emb, _category_name_embeddings[category]).item()
       
        combined = 0.4 * content_score + 0.6 * name_score

        if combined > best_score:
            best_score = combined
            best_category = category

    if best_score < 0.38:
        return "__clarify__"

    return best_category