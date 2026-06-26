import os
import json
import math
import joblib
import numpy as np
import ir_datasets
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ✅ الدالة المفقودة! لازم تكون موجودة لأن TF-IDF محفوظ معها
def identity_analyzer(doc):
    """دالة تحليل تأخذ الـ tokens المحفوظة كما هي دون إعادة تقسيم"""
    return doc

app = FastAPI(title="Evaluation Service (Port 8005)")

MODELS_DIR = r"C:\Users\User\IR_Project\models"
HISTORY_DIR = r"C:\Users\User\IR_Project\history"  # ✅ للـ Personalization

DATASET_IDS = {
    "cord19": "cord19/trec-covid",
    "clinical": "clinicaltrials/2021/trec-ct-2021"
}

class EvalRequest(BaseModel):
    dataset_name: str
    model_type: str
    top_k: int = 100
    k1: Optional[float] = 1.5
    b: Optional[float] = 0.75
    max_queries: Optional[int] = 0
    use_personalization: bool = False  # ✅ جديد
    user_id: str = "user1"  # ✅ جديد

class EvaluationService:
    def __init__(self):
        self._cache = {}
        self.embedding_model = None

    def _get_embedding_model(self):
        if self.embedding_model is None:
            print("[*] Loading SentenceTransformer for evaluation...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedding_model

    def _load_bm25(self, index_name, k1, b):
        key = f"{index_name}_bm25_{k1}_{b}"
        if key not in self._cache:
            path = os.path.join(MODELS_DIR, f"{index_name}_bm25.pkl")
            data = joblib.load(path)
            bm25 = BM25Okapi(data['corpus_tokens'], k1=k1, b=b)
            self._cache[key] = (bm25, data['doc_ids'])
        return self._cache[key]

    def _load_tfidf(self, index_name):
        key = f"{index_name}_tfidf"
        if key not in self._cache:
            path = os.path.join(MODELS_DIR, f"{index_name}_tfidf.pkl")
            # ✅ joblib سيبحث عن identity_analyzer في هذا الملف الآن
            data = joblib.load(path)
            self._cache[key] = (data['vectorizer'], data['tfidf_matrix'], data['doc_ids'])
        return self._cache[key]

    def _load_embeddings(self, index_name):
        key = f"{index_name}_emb"
        if key not in self._cache:
            path = os.path.join(MODELS_DIR, f"{index_name}_embeddings.pkl")
            data = joblib.load(path)
            self._cache[key] = (data['embeddings'], data['doc_ids'])
        return self._cache[key]

    def _search_bm25(self, query_tokens, index_name, top_k, k1, b):
        bm25, doc_ids = self._load_bm25(index_name, k1, b)
        scores = bm25.get_scores(query_tokens)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [str(doc_ids[i]) for i in top_indices if scores[i] > 0]

    def _search_tfidf(self, query_tokens, index_name, top_k):
        vectorizer, matrix, doc_ids = self._load_tfidf(index_name)
        q = vectorizer.transform([query_tokens])
        scores = (matrix * q.T).toarray().flatten()
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [str(doc_ids[i]) for i in top_indices if scores[i] > 0]

    def _search_embedding(self, query_text, index_name, top_k):
        embeddings, doc_ids = self._load_embeddings(index_name)
        model = self._get_embedding_model()
        query_emb = model.encode(query_text, convert_to_numpy=True)
        scores = np.dot(embeddings, query_emb) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
        )
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [str(doc_ids[i]) for i in top_indices]

    def _search_hybrid_serial(self, query_tokens, query_text, index_name, top_k, k1, b):
        tfidf_results = self._search_tfidf(query_tokens, index_name, 200)
        if not tfidf_results:
            return []
        embeddings, doc_ids = self._load_embeddings(index_name)
        model = self._get_embedding_model()
        query_emb = model.encode(query_text, convert_to_numpy=True)
        doc_id_to_idx = {str(did): i for i, did in enumerate(doc_ids)}
        candidate_indices = [doc_id_to_idx[did] for did in tfidf_results if did in doc_id_to_idx]
        if not candidate_indices:
            return tfidf_results[:top_k]
        candidate_embeddings = embeddings[candidate_indices]
        scores = np.dot(candidate_embeddings, query_emb) / (
            np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
        )
        top_local = np.argsort(scores)[::-1][:top_k]
        return [str(doc_ids[candidate_indices[i]]) for i in top_local]

    def _search_hybrid_parallel(self, query_tokens, query_text, index_name, top_k, k1, b):
        tfidf_results = self._search_tfidf(query_tokens, index_name, 50)
        emb_results = self._search_embedding(query_text, index_name, 50)
        k_rrf = 60
        rrf_scores = {}
        for rank, doc_id in enumerate(tfidf_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
        for rank, doc_id in enumerate(emb_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [doc_id for doc_id, _ in sorted_docs]

    def _tokenize(self, text):
        if not text or not isinstance(text, str) or not text.strip():
            return []
        return text.lower().split()

    # ==========================================
    # ✅ Personalization - Query Expansion (ذكي حسب الداتاست)
    # ==========================================
    def _apply_personalization(self, query: str, user_id: str, dataset_name: str) -> str:
        """إضافة keywords من تاريخ البحث للـ query - فقط من نفس الداتاست"""
        history_file = os.path.join(HISTORY_DIR, f"{user_id}.json")
        
        if not os.path.exists(history_file):
            print(f"[!] No history found for user: {user_id}")
            return query
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # ✅ فلتر حسب الداتاست
            filtered_history = [
                entry for entry in history 
                if entry.get("dataset_name") == dataset_name
            ]
            
            if not filtered_history:
                print(f"[!] No history for dataset: {dataset_name}")
                return query
            
            word_counts = {}
            for entry in filtered_history:
                for word in entry.get("query", "").split():
                    w = word.lower().strip()
                    if len(w) >= 3:
                        word_counts[w] = word_counts.get(w, 0) + 1
            
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            preferred_topics = [w[0] for w in top_words]
            
            if preferred_topics:
                expanded_query = query + " " + " ".join(preferred_topics)
                print(f"[*] Personalization ({dataset_name}): '{query[:50]}...' → '{expanded_query[:80]}...'")
                return expanded_query
            
            return query
        except Exception as e:
            print(f"[!] Personalization error: {e}")
            return query

    def get_search_results(self, query, dataset_name, model_type, top_k, k1, b,
                          use_personalization=False, user_id="user1"):
        # ✅ تطبيق Personalization قبل البحث
        if use_personalization:
            query = self._apply_personalization(query, user_id, dataset_name)  # ✅ أضفنا dataset_name
        
        tokens = self._tokenize(query)
        
        if model_type == "bm25":
            return self._search_bm25(tokens, dataset_name, top_k, k1, b)
        elif model_type == "tfidf":
            return self._search_tfidf(tokens, dataset_name, top_k)
        elif model_type == "embedding":
            return self._search_embedding(query, dataset_name, top_k)
        elif model_type == "hybrid_serial":
            return self._search_hybrid_serial(tokens, query, dataset_name, top_k, k1, b)
        elif model_type == "hybrid_parallel":
            return self._search_hybrid_parallel(tokens, query, dataset_name, top_k, k1, b)
        else:
            raise ValueError(f"Model type '{model_type}' not supported")

    def precision_at_k(self, retrieved, relevant, k):
        retrieved_k = retrieved[:k]
        return sum(1 for doc in retrieved_k if doc in relevant) / k if retrieved_k else 0.0

    def recall_at_k(self, retrieved, relevant, k):
        if not relevant:
            return 0.0
        retrieved_k = retrieved[:k]
        return sum(1 for doc in retrieved_k if doc in relevant) / len(relevant)

    def average_precision(self, retrieved, relevant):
        if not relevant:
            return 0.0
        hits, sum_p = 0, 0.0
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                hits += 1
                sum_p += hits / (i + 1)
        return sum_p / len(relevant)

    def ndcg_at_k(self, retrieved, relevant, k):
        retrieved_k = retrieved[:k]
        dcg = sum(1 / math.log2(i + 2) for i, doc in enumerate(retrieved_k) if doc in relevant)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate(self, dataset_name, model_type, top_k, k1, b, max_queries,
                use_personalization=False, user_id="user1"):
        dataset_id = DATASET_IDS.get(dataset_name)
        if not dataset_id:
            raise ValueError(f"Dataset غير مدعوم: {dataset_name}")

        print(f"[*] Loading dataset: {dataset_id}")
        dataset = ir_datasets.load(dataset_id)
        
        qrels = {}
        for qrel in dataset.qrels_iter():
            if qrel.relevance > 0:
                qrels.setdefault(qrel.query_id, set()).add(str(qrel.doc_id))
        
        print(f"[✓] Loaded {len(qrels)} queries with qrels")

        queries = {}
        for query in dataset.queries_iter():
            if query.query_id in qrels:
                queries[query.query_id] = (query.title + " " + query.description) if dataset_name == "cord19" else query.text

        print(f"[*] Total queries available: {len(queries)}")

        if max_queries and max_queries > 0:
            queries = dict(list(queries.items())[:max_queries])

        print(f"[*] Evaluating on {len(queries)} queries")
        print(f"[*] Personalization: {'ENABLED' if use_personalization else 'DISABLED'}")

        if not queries:
            raise ValueError("لا توجد استعلامات للتقييم")

        map_scores, recall_scores, precision_scores, ndcg_scores = [], [], [], []
        per_query_results = []

        for idx, (query_id, query_text) in enumerate(queries.items(), 1):
            if idx % 10 == 0 or idx == len(queries):
                print(f"  [{idx}/{len(queries)}] Query {query_id}...")
            
            relevant = qrels.get(query_id, set())
            retrieved = self.get_search_results(
                query_text, dataset_name, model_type, top_k, k1, b,
                use_personalization=use_personalization,
                user_id=user_id
            )

            ap = self.average_precision(retrieved, relevant)
            rec = self.recall_at_k(retrieved, relevant, top_k)
            prec = self.precision_at_k(retrieved, relevant, 10)
            ndcg = self.ndcg_at_k(retrieved, relevant, 10)

            map_scores.append(ap)
            recall_scores.append(rec)
            precision_scores.append(prec)
            ndcg_scores.append(ndcg)

            per_query_results.append({
                "query_id": query_id,
                "query_text": query_text[:150],
                "AP": round(ap, 4),
                "Recall": round(rec, 4),
                "Precision@10": round(prec, 4),
                "nDCG": round(ndcg, 4),
                "retrieved_count": len(retrieved),
                "relevant_count": len(relevant),
                "retrieved_docs": retrieved[:30],
                "relevant_docs": list(relevant)[:30]
            })

        return {
            "dataset": dataset_name, 
            "model_type": model_type, 
            "top_k": top_k,
            "num_queries_evaluated": len(map_scores),
            "total_queries_in_dataset": len(qrels),
            "use_personalization": use_personalization,
            "user_id": user_id,
            "metrics": {
                "MAP": round(sum(map_scores) / len(map_scores), 4) if map_scores else 0,
                "Recall": round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0,
                "Precision@10": round(sum(precision_scores) / len(precision_scores), 4) if precision_scores else 0,
                "nDCG": round(sum(ndcg_scores) / len(ndcg_scores), 4) if ndcg_scores else 0
            },
            "per_query_results": per_query_results
        }

evaluator = EvaluationService()

@app.post("/evaluate")
async def evaluate_endpoint(request: EvalRequest):
    try:
        return {"status": "success", **evaluator.evaluate(**request.model_dump())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005, reload=False)