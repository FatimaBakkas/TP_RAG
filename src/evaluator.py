# src/evaluator.py
# src/evaluator.py
from src.embedder import Embedder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import yaml

def evaluate_answer(generated_answer, reference_answer, config_file="config.yaml"):
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    embedder = Embedder(model_name=config["embedding_model"])
    gen_emb = embedder.embed_query(generated_answer)
    ref_emb = embedder.embed_query(reference_answer)
    sim_score = cosine_similarity([gen_emb], [ref_emb])[0][0]
    return np.round(sim_score, 4)
