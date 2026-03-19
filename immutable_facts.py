import pickle
import torch
from sentence_transformers import util

FACTS_PATH = "immutable_facts.pkl"
MATCH_THRESHOLD = 0.75  

_fact_sentences = []
_fact_embeddings = None
_loaded = False


def load_facts(embedder):
   
    global _fact_sentences, _fact_embeddings, _loaded

    if _loaded:
        return

    try:
        with open(FACTS_PATH, "rb") as f:
            all_facts = pickle.load(f)

        for category, facts in all_facts.items():
            for sent, score in facts:
                _fact_sentences.append(sent)

        if _fact_sentences:
            _fact_embeddings = embedder.encode(
                _fact_sentences, convert_to_tensor=True, show_progress_bar=False
            )
            print(f"Loaded {len(_fact_sentences)} immutable facts.")
        else:
            print("Warning: no immutable facts found.")

    except FileNotFoundError:
        print("Warning: immutable_facts.pkl not found. Run extract_immutable.py first.")

    _loaded = True


def check_immutable(question, embedder):
    if not _fact_sentences or _fact_embeddings is None:
        return None

    q_emb = embedder.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, _fact_embeddings)[0].tolist()

    best_idx = int(torch.tensor(scores).argmax().item())
    best_score = scores[best_idx]

    if best_score >= MATCH_THRESHOLD:
        return _fact_sentences[best_idx]

    return None