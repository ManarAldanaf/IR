import os
import joblib
import numpy as np
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def identity_analyzer(doc):
    return doc
    
app = FastAPI(title="Retrieval Service (Port 8003)")

PREPROCESSING_SERVICE_URL = "http://127.0.0.1:8001/clean-text"
DOCUMENT_SERVICE_URL = "http://127.0.0.1:8006/get-documents"
MODELS_DIR = r"C:\Users\User\IR_Project\models"
HISTORY_DIR = r"C:\Users\User\IR_Project\history"  # ✅ جديد - للـ Personalization

# ✅ كاش عالمي — كل الفهارس بتتحمل هون مرة وحدة بس
GLOBAL_CACHE = {}

# ✅ الداتاسيتات والنماذج اللي بدنا نحمّلها مسبقاً
DATASETS_TO_PRELOAD = ["cord19", "clinical"]

class SearchRequest(BaseModel):
    dataset_name: str
    query: str
    model_type: str
    top_k: int = 10
    k1: Optional[float] = 1.5
    b: Optional[float] = 0.75
    fetch_text: bool = True
    use_personalization: bool = False  # ✅ جديد - الطلب الإضافي (البند 16)
    user_id: str = "user1"  # ✅ جديد

class RetrievalStrategy:
    def search(self, query_tokens: List[str], top_k: int) -> List[Dict[str, Any]]:
        raise NotImplementedError

class BM25Strategy(RetrievalStrategy):
    def __init__(self, dataset_name: str, k1: float = 1.5, b: float = 0.75):
        self.dataset_name = dataset_name
        self.k1 = k1
        self.b = b
        cache_key = f"{dataset_name}_bm25_{k1}_{b}"
        if cache_key not in GLOBAL_CACHE:
            index_path = os.path.join(MODELS_DIR, f"{dataset_name}_bm25.pkl")
            data = joblib.load(index_path)
            GLOBAL_CACHE[cache_key] = {
                "doc_ids": data["doc_ids"],
                "bm25": BM25Okapi(data["corpus_tokens"], k1=k1, b=b)
            }
        self.doc_ids = GLOBAL_CACHE[cache_key]["doc_ids"]
        self.bm25_model = GLOBAL_CACHE[cache_key]["bm25"]

    def search(self, query_tokens: List[str], top_k: int) -> List[Dict[str, Any]]:
        if not query_tokens:
            return []
        scores = self.bm25_model.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"doc_id": str(self.doc_ids[i]), "score": float(scores[i]), "rank": rank + 1}
                for rank, i in enumerate(top_indices) if scores[i] > 0]

class TFIDFStrategy(RetrievalStrategy):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        cache_key = f"{dataset_name}_tfidf"
        if cache_key not in GLOBAL_CACHE:
            index_path = os.path.join(MODELS_DIR, f"{dataset_name}_tfidf.pkl")
            GLOBAL_CACHE[cache_key] = joblib.load(index_path)
        data = GLOBAL_CACHE[cache_key]
        self.vectorizer = data['vectorizer']
        self.tfidf_matrix = data['tfidf_matrix']
        self.doc_ids = data['doc_ids']

    def search(self, query_tokens: List[str], top_k: int) -> List[Dict[str, Any]]:
        if not query_tokens:
            return []
        query_vector = self.vectorizer.transform([query_tokens])
        similarity_scores = (self.tfidf_matrix * query_vector.T).toarray().flatten()
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
        return [{"doc_id": str(self.doc_ids[i]), "score": float(similarity_scores[i]), "rank": rank + 1}
                for rank, i in enumerate(top_indices) if similarity_scores[i] > 0]

class EmbeddingStrategy(RetrievalStrategy):
    def __init__(self, dataset_name: str, model_name: str = 'all-MiniLM-L6-v2'):
        if 'encoder' not in GLOBAL_CACHE:
            GLOBAL_CACHE['encoder'] = SentenceTransformer(model_name)
        self.model = GLOBAL_CACHE['encoder']
        cache_key = f"{dataset_name}_emb"
        if cache_key not in GLOBAL_CACHE:
            index_path = os.path.join(MODELS_DIR, f"{dataset_name}_embeddings.pkl")
            GLOBAL_CACHE[cache_key] = joblib.load(index_path)
        data = GLOBAL_CACHE[cache_key]
        self.embeddings = data['embeddings']
        self.doc_ids = data['doc_ids']

    def search(self, query_tokens: List[str], top_k: int) -> List[Dict[str, Any]]:
        if not query_tokens:
            return []
        query_emb = self.model.encode(' '.join(query_tokens), convert_to_numpy=True)
        scores = np.dot(self.embeddings, query_emb) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"doc_id": str(self.doc_ids[i]), "score": float(scores[i]), "rank": rank + 1}
                for rank, i in enumerate(top_indices)]

class HybridSerialStrategy(RetrievalStrategy):
    def __init__(self, dataset_name: str):
        self.tfidf = TFIDFStrategy(dataset_name)
        self.embed = EmbeddingStrategy(dataset_name)

    def search(self, query_tokens: List[str], top_k: int) -> List[Dict[str, Any]]:
        candidates = self.tfidf.search(query_tokens, top_k=200)
        c_ids = set(c['doc_id'] for c in candidates)
        embed_results = self.embed.search(query_tokens, top_k=200)
        reranked = [r for r in embed_results if r['doc_id'] in c_ids][:top_k]
        for rank, item in enumerate(reranked):
            item['rank'] = rank + 1
        return reranked

class HybridParallelStrategy(RetrievalStrategy):
    def __init__(self, dataset_name: str):
        self.tfidf = TFIDFStrategy(dataset_name)
        self.embed = EmbeddingStrategy(dataset_name)

    def search(self, query_tokens: List[str], top_k: int) -> List[Dict[str, Any]]:
        tfidf_results = self.tfidf.search(query_tokens, top_k=100)
        embed_results = self.embed.search(query_tokens, top_k=100)
        rrf_k = 60
        rrf_scores = {}
        for rank, r in enumerate(tfidf_results):
            rrf_scores[r['doc_id']] = rrf_scores.get(r['doc_id'], 0) + 1 / (rrf_k + rank + 1)
        for rank, r in enumerate(embed_results):
            rrf_scores[r['doc_id']] = rrf_scores.get(r['doc_id'], 0) + 1 / (rrf_k + rank + 1)
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"doc_id": doc_id, "score": round(score, 6), "rank": rank + 1}
                for rank, (doc_id, score) in enumerate(sorted_docs)]

class RetrievalServiceFactory:
    def __init__(self):
        self.strategies = {}

    def get_strategy(self, name, m_type, k1=1.5, b=0.75):
        key = f"{name}_{m_type}_{k1}_{b}" if m_type == "bm25" else f"{name}_{m_type}"
        if key not in self.strategies:
            if m_type == "bm25":
                self.strategies[key] = BM25Strategy(name, k1=k1, b=b)
            elif m_type == "tfidf":
                self.strategies[key] = TFIDFStrategy(name)
            elif m_type == "embedding":
                self.strategies[key] = EmbeddingStrategy(name)
            elif m_type == "hybrid_serial":
                self.strategies[key] = HybridSerialStrategy(name)
            elif m_type == "hybrid_parallel":
                self.strategies[key] = HybridParallelStrategy(name)
            else:
                raise ValueError(f"Model type '{m_type}' not supported")
        return self.strategies[key]

factory = RetrievalServiceFactory()

# ============================================================
# ✅ PRE-LOADING — هاد الجزء الأهم
# يحمّل كل الفهارس بالذاكرة فوراً عند تشغيل uvicorn
# قبل ما يجي أي مستخدم يسأل query
# ============================================================
@app.on_event("startup")
async def preload_all_indices():
    print("=" * 60)
    print("[*] جاري تحميل كل الفهارس بالذاكرة (Pre-loading)...")
    print("=" * 60)

    for dataset_name in DATASETS_TO_PRELOAD:
        try:
            print(f"\n[*] تحميل فهارس: {dataset_name}")

            print(f"    -> BM25...")
            factory.get_strategy(dataset_name, "bm25")

            print(f"    -> TF-IDF...")
            factory.get_strategy(dataset_name, "tfidf")

            print(f"    -> Embedding...")
            factory.get_strategy(dataset_name, "embedding")

            print(f"    -> Hybrid Serial...")
            factory.get_strategy(dataset_name, "hybrid_serial")

            print(f"    -> Hybrid Parallel...")
            factory.get_strategy(dataset_name, "hybrid_parallel")

            print(f"[✓] {dataset_name} — كل الفهارس جاهزة بالذاكرة")

        except FileNotFoundError as e:
            print(f"[!] تحذير: فهرس غير موجود لـ {dataset_name}: {e}")
        except Exception as e:
            print(f"[!] خطأ بتحميل {dataset_name}: {e}")

    print("\n" + "=" * 60)
    print("[✓✓✓] كل الفهارس محمّلة بالذاكرة! النظام جاهز للاستعلامات السريعة")
    print("=" * 60 + "\n")

# ============================================================
# ✅ Personalization - الطلب الإضافي (البند 16)
# ============================================================
def apply_personalization(results: List[Dict[str, Any]], user_id: str, dataset_name: str, top_k: int) -> Dict:
    """
    تطبيق Personalization:
    1. قراءة سجل البحث من history/
    2. فلتر حسب الداتاست (فقط من نفس الداتاست)
    3. استخراج الكلمات الأكثر تكراراً (user profile)
    4. إعطاء boost للنتائج المتعلقة بها
    """
    user_profile = {
        "user_id": user_id,
        "preferred_topics": [],
        "search_count": 0
    }
    
    history_file = os.path.join(HISTORY_DIR, f"{user_id}.json")
    
    if not os.path.exists(history_file):
        return {"results": results, "user_profile": user_profile}
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # ✅ فلتر حسب الداتاست (فقط من نفس الداتاست)
        filtered_history = [e for e in history if e.get("dataset_name") == dataset_name]
        user_profile["search_count"] = len(filtered_history)
        
        if not filtered_history:
            return {"results": results, "user_profile": user_profile}
        
        # استخراج الكلمات الأكثر تكراراً
        word_counts = {}
        for entry in filtered_history:
            for word in entry.get("query", "").split():
                w = word.lower().strip()
                if len(w) >= 3:
                    word_counts[w] = word_counts.get(w, 0) + 1
        
        # أهم 10 كلمات
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        preferred_topics = [w[0] for w in top_words]
        user_profile["preferred_topics"] = preferred_topics
        
        # ✅ تعديل ترتيب النتائج (boost)
        if preferred_topics and results:
            for result in results:
                doc_id = result["doc_id"].lower()
                # حساب boost بناءً على عدد الكلمات المطابقة
                boost = sum(0.15 for topic in preferred_topics if topic in doc_id)
                result["score"] = result["score"] * (1 + boost)
            
            # إعادة الترتيب حسب score الجديد
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]
            
            # تحديث rank
            for rank, result in enumerate(results, 1):
                result["rank"] = rank
        
    except Exception as e:
        print(f"[!] Personalization error: {e}")
    
    return {"results": results, "user_profile": user_profile}

# ============================================================
# Search Endpoint
# ============================================================
@app.post("/search")
async def search_endpoint(request: SearchRequest):
    try:
        try:
            res = requests.post(PREPROCESSING_SERVICE_URL, json={"text": request.query}, timeout=5)
            tokens = res.json().get("tokens", []) if res.status_code == 200 else request.query.lower().split()
        except:
            tokens = request.query.lower().split()

        strategy = factory.get_strategy(request.dataset_name, request.model_type, k1=request.k1 or 1.5, b=request.b or 0.75)
        results = strategy.search(tokens, request.top_k * 2)  # نجيب ضعف العدد للترتيب

        # ✅ Personalization (الطلب الإضافي - البند 16)
        user_profile = None
        if request.use_personalization:
            pers_result = apply_personalization(results, request.user_id, request.dataset_name, request.top_k)
            results = pers_result["results"]
            user_profile = pers_result["user_profile"]
        else:
            results = results[:request.top_k]
            for rank, result in enumerate(results, 1):
                result["rank"] = rank

        if request.fetch_text:
            try:
                doc_ids = [r["doc_id"] for r in results]
                dr = requests.post(
                    DOCUMENT_SERVICE_URL,
                    json={"doc_ids": doc_ids, "dataset_name": request.dataset_name},
                    timeout=10
                )
                if dr.status_code == 200:
                    documents = dr.json().get("documents", {})
                    for r in results:
                        r["text"] = documents.get(r["doc_id"], "")
            except Exception as e:
                print(f"DB error: {e}")

        response_data = {
            "status": "success",
            "dataset": request.dataset_name,
            "model": request.model_type,
            "query": request.query,
            "results": results
        }
        
        # ✅ إضافة user_profile للـ Response إذا كان Personalization مفعل
        if user_profile:
            response_data["user_profile"] = user_profile
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    loaded = list(GLOBAL_CACHE.keys())
    return {"status": "running", "preloaded_count": len(loaded)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003, reload=False)