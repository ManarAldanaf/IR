import os
import json
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(title="Indexing Service (Port 8002)")

def identity_analyzer(doc):
    """دالة تحليل تأخذ الـ tokens المحفوظة كما هي دون إعادة تقسيم"""
    return doc

class IndexRequest(BaseModel):
    dataset_name: str
    build_embeddings: bool = False

class IndexingService:
    def __init__(self):
        self.data_dir = r"C:\Users\User\IR_Project\datasets"
        self.indices_dir = r"C:\Users\User\IR_Project\models"
        os.makedirs(self.indices_dir, exist_ok=True)
        self.embedding_model = None

    def load_embeddings_model(self):
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            print("[*] Loading embeddings model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def build_all_indices(self, dataset_name, build_embeddings=False):
        file_path = os.path.join(self.data_dir, f"{dataset_name}_processed.jsonl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing: {file_path}")

        doc_ids, corpus_tokens, corpus_texts = [], [], []
        print(f"[*] Reading dataset: {dataset_name}...")
        
        with open(file_path, "r", encoding="utf8") as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line)
                doc_ids.append(str(data["doc_id"]))
                corpus_tokens.append(data.get("tokens", []))
                corpus_texts.append(data.get("text", ""))
                
                if line_num % 50000 == 0:
                    print(f"  [*] Read {line_num} documents...")

        if not doc_ids:
            raise ValueError("Empty dataset")

        print(f"[✓] Total documents: {len(doc_ids)}")

        # ==================
        # 1. BM25
        # ==================
        print("[*] Building BM25...")
        bm25_data = {"doc_ids": doc_ids, "corpus_tokens": corpus_tokens}
        joblib.dump(bm25_data, os.path.join(self.indices_dir, f"{dataset_name}_bm25.pkl"), compress=3)
        print("[✓] BM25 done")

        # ==================
        # 2. TF-IDF
        # ==================
        print("[*] Building TF-IDF...")
        vectorizer = TfidfVectorizer(
            max_features=50000,
            analyzer=identity_analyzer
        )
        tfidf_matrix = vectorizer.fit_transform(corpus_tokens)

        tfidf_data = {
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "doc_ids": doc_ids
        }
        joblib.dump(tfidf_data, os.path.join(self.indices_dir, f"{dataset_name}_tfidf.pkl"), compress=3)
        print("[✓] TF-IDF done")

        # ==================
        # 3. EMBEDDINGS
        # ==================
        if build_embeddings:
            self.load_embeddings_model()
            print("[*] Building embeddings...")
            emb = self.embedding_model.encode(corpus_texts, batch_size=64, show_progress_bar=True)

            emb_data = {
                "embeddings": emb,
                "doc_ids": doc_ids
            }
            joblib.dump(emb_data, os.path.join(self.indices_dir, f"{dataset_name}_embeddings.pkl"), compress=3)
            print("[✓] Embeddings done")

        print("\n[SUCCESS] All indices saved with compression (compress=3).")

indexer = IndexingService()

@app.post("/build-index")
async def build_index(request: IndexRequest):
    try:
        indexer.build_all_indices(request.dataset_name, request.build_embeddings)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002, reload=False)