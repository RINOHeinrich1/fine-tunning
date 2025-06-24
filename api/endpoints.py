from fastapi import APIRouter, HTTPException
from .schemas import QuestionRequest, FeedbackRequest
from model.embedder import load_model
from model.faiss_index import build_faiss_index, search
from model.documents import get_documents
from model.fine_tuning import fine_tune_until_margin_respected
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from model.document_parser import extract_text
from model.embedding import get_embedding, chunk_text_optimale
import numpy as np
from config import BATCH_SIZE, EPOCHS, WARMUP_STEPS, DEVICE
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import os
import uuid
import shutil
import tempfile
from fastapi.responses import FileResponse
from dotenv import load_dotenv
load_dotenv()
router = APIRouter()

model = load_model()
documents = get_documents()
doc_embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
index = build_faiss_index(doc_embeddings)
CHATBOT_RELOAD_URL = os.getenv("CHATBOT_RELOAD_URL", "http://localhost:8000/reload-model")  # À adapter selon ton infra


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("COLLECTION_NAME")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))

class DeployRequest(BaseModel):
    version: str = "esti-rag-ft"

def get_next_model_version(base_name="esti-rag-ft", models_dir="./models") -> str:
    existing_versions = []
    for name in os.listdir(models_dir):
        if name.startswith(base_name + "-v"):
            try:
                version_num = int(name.replace(base_name + "-v", ""))
                existing_versions.append(version_num)
            except ValueError:
                continue
    next_version = max(existing_versions + [0]) + 1
    return f"{base_name}-v{next_version}"

@router.get("/")
def root():
    return {"message": "✅ RAG Webservice is running."}

@router.get("/documents")
def list_documents():
    return {"documents": documents}

@router.post("/ask")
def ask(request: QuestionRequest):
    try:
        results, scores = search(index, model, documents, request.question, request.top_k)
        return {
            "question": request.question,
            "results": [{"doc": d, "score": s} for d, s in zip(results, scores)]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
def feedback(request: FeedbackRequest):
    global model, doc_embeddings, index

    try:
        before_docs, before_scores = search(index, model, documents, request.question)
        model = fine_tune_until_margin_respected(
            request.question, request.positive_docs, request.negative_docs,
            model, BATCH_SIZE, EPOCHS, WARMUP_STEPS, DEVICE,30
        )
        doc_embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
        index = build_faiss_index(doc_embeddings)
        after_docs, after_scores = search(index, model, documents, request.question)

        return {
            "message": "✅ Fine-tuning terminé et index mis à jour.",
            "comparison": {
                "before": [{"doc": d, "score": s} for d, s in zip(before_docs, before_scores)],
                "after": [{"doc": d, "score": s} for d, s in zip(after_docs, after_scores)]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deploy")
def deploy_model():
    try:
        # 1. Générer une version
        version_name = get_next_model_version()

        model_src_dir = f"./models/esti-rag-ft"
        model_version_dir = f"./models/{version_name}"
        zip_path = f"./exported/{version_name}.zip"

        # 2. Copier le dossier du modèle vers une nouvelle version
        shutil.copytree(model_src_dir, model_version_dir)

        # 3. Créer dossier exporté s’il n’existe pas
        os.makedirs("./exported", exist_ok=True)

        # 4. Zipper la nouvelle version
        if os.path.exists(zip_path):
            os.remove(zip_path)
        shutil.make_archive(f"./exported/{version_name}", 'zip', model_version_dir)

        # 5. Construire l’URL de téléchargement
        model_url = f"http://localhost:8001/download-model?version={version_name}"

        # 6. Notifier le chatbot-service
        response = requests.get(CHATBOT_RELOAD_URL, params={
            "version": version_name,
            "url": model_url
        })

        if response.status_code != 200:
            raise Exception(f"Erreur de notification: {response.text}")

        return {
            "message": f"✅ Modèle '{version_name}' exporté et notification envoyée",
            "chatbot_response": response.json()
        }

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download-model")
def download_model(version: str):
    zip_path = f"./exported/{version}.zip"
    if not os.path.isfile(zip_path):
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    return FileResponse(path=zip_path, filename=f"{version}.zip", media_type="application/zip")


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Initialiser la collection
client.recreate_collection(
    COLLECTION,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance="Cosine"),
)

@router.post("/upload-file")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    filepath = f"/tmp/{file.filename}"
    with open(filepath, "wb") as f:
        f.write(contents)

    text = extract_text(filepath)
    os.remove(filepath)

    points = []
    for chunk in  chunk_text_optimale(text):
        embedding = get_embedding(chunk)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk, "source": file.filename}
        ))

    client.upsert(collection_name=COLLECTION, points=points)
    return {"status": "ok", "chunks": len(points)}

@router.get("/search-docs")
def search(q: str):
    vector = get_embedding(q)
    results = client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=5,
        with_payload=True
    )
    return [{"text": r.payload["text"], "score": r.score} for r in results]