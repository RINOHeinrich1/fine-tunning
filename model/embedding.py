from sentence_transformers import SentenceTransformer
import os
import torch
import numpy as np
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' #"dangvantuan/sentence-camembert-large"
MODELS_DIR = "./models"

def get_latest_model_path():
    """
    Renvoie le chemin du modèle versionné le plus récent dans ./models,
    ou DEFAULT_MODEL_NAME si aucun modèle local n’est trouvé.
    """
    if not os.path.exists(MODELS_DIR):
        return DEFAULT_MODEL_NAME

    versions = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d)) and d.startswith("esti-rag-ft-v")]
    if not versions:
        return DEFAULT_MODEL_NAME

    # Trier par numéro de version décroissant
    versions.sort(key=lambda v: int(v.split("-v")[-1]), reverse=True)
    return os.path.join(MODELS_DIR, versions[0])

# Chargement du modèle SentenceTransformer
MODEL_PATH = get_latest_model_path()
print(f"📦 Utilisation du modèle : {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH, device=DEVICE)

def get_embedding(texts):
    model_path = get_latest_model_path()
    print(f"🧠 Chargement SentenceTransformer depuis : {model_path}")
    model = SentenceTransformer(model_path, device=DEVICE)
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def chunk_text_optimale(text: str):
    sections = re.split(r"(🌐|🧩|👥|📈|🎯|📞)", text)
    chunks = []
    
    i = 1
    while i < len(sections):
        emoji = sections[i]
        content = sections[i + 1].strip()
        # Ajouter le titre emoji + contenu
        if emoji and content:
            chunks.append(f"{emoji} {content}")
        i += 2

    return chunks