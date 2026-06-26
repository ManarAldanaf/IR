from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx

app = FastAPI(title="IR Project 2026 - API Gateway")

PREPROCESSING_SERVICE_URL    = "http://127.0.0.1:8001"
INDEXING_SERVICE_URL         = "http://127.0.0.1:8002"
RETRIEVAL_SERVICE_URL        = "http://127.0.0.1:8003"
QUERY_REFINEMENT_SERVICE_URL = "http://127.0.0.1:8004"
EVALUATION_SERVICE_URL       = "http://127.0.0.1:8005"

TIMEOUT = httpx.Timeout(300.0)

class TextRequest(BaseModel):
    text: str

class IndexRequest(BaseModel):
    dataset_name: str

class SearchRequest(BaseModel):
    dataset_name: str
    query: str
    model_type: str
    top_k: int = 10
    k1: Optional[float] = 1.5
    b: Optional[float] = 0.75
    use_refinement: Optional[bool] = False
    use_personalization: Optional[bool] = False
    user_id: Optional[str] = "user1"

class RefineRequest(BaseModel):
    query: str
    dataset_name: str
    user_id: Optional[str] = "default"
    use_spell_check: Optional[bool] = True
    use_synonyms: Optional[bool] = True
    use_history: Optional[bool] = True

class EvalRequest(BaseModel):
    dataset_name: str
    model_type: str
    top_k: int = 10
    k1: Optional[float] = 1.5
    b: Optional[float] = 0.75
    max_queries: Optional[int] = 50
    use_personalization: Optional[bool] = False  # ✅ جديد
    user_id: Optional[str] = "user1"  # ✅ جديد

@app.get("/")
async def root():
    return {"message": "API Gateway is running!", "version": "2026"}

@app.get("/health")
async def health():
    services = {
        "preprocessing":    f"{PREPROCESSING_SERVICE_URL}/docs",
        "indexing":         f"{INDEXING_SERVICE_URL}/docs",
        "retrieval":        f"{RETRIEVAL_SERVICE_URL}/docs",
        "query_refinement": f"{QUERY_REFINEMENT_SERVICE_URL}/docs",
        "evaluation":       f"{EVALUATION_SERVICE_URL}/docs",
    }
    status = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        for name, url in services.items():
            try:
                r = await client.get(url)
                status[name] = "up" if r.status_code == 200 else "error"
            except Exception:
                status[name] = "down"
    return {"services": status}

@app.post("/api/v1/preprocess")
async def preprocess(request: TextRequest):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.post(f"{PREPROCESSING_SERVICE_URL}/clean-text", json={"text": request.text})
            r.raise_for_status()
            return r.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Preprocessing service unavailable: {e}")

@app.post("/api/v1/build-index")
async def build_index(request: IndexRequest):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.post(f"{INDEXING_SERVICE_URL}/build-index", json={"dataset_name": request.dataset_name})
            r.raise_for_status()
            return r.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Indexing service unavailable: {e}")

@app.post("/api/v1/refine-query")
async def refine_query(request: RefineRequest):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.post(f"{QUERY_REFINEMENT_SERVICE_URL}/api/v1/refine-query", json={
                "query": request.query,
                "dataset_name": request.dataset_name,
                "user_id": request.user_id,
                "use_spell_check": request.use_spell_check,
                "use_synonyms": request.use_synonyms,
                "use_history": request.use_history
            })
            r.raise_for_status()
            return r.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Query refinement service unavailable: {e}")

@app.post("/api/v1/search")
async def search(request: SearchRequest):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            query = request.query
            
            if request.use_refinement:
                r = await client.post(f"{QUERY_REFINEMENT_SERVICE_URL}/api/v1/refine-query", json={
                    "query": query,
                    "dataset_name": request.dataset_name,
                    "user_id": request.user_id,
                    "use_spell_check": True,
                    "use_synonyms": True,
                    "use_history": True
                })
                if r.status_code == 200:
                    query = r.json().get("refined_query", query)
            
            r = await client.post(f"{RETRIEVAL_SERVICE_URL}/search", json={
                "dataset_name": request.dataset_name,
                "query": query,
                "model_type": request.model_type,
                "top_k": request.top_k,
                "k1": request.k1,
                "b": request.b,
                "fetch_text": True,
                "use_personalization": request.use_personalization,
                "user_id": request.user_id
            })
            r.raise_for_status()
            result = r.json()
            result["used_query"] = query
            result["refinement_applied"] = request.use_refinement
            result["personalization_applied"] = request.use_personalization
            return result
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")

@app.post("/api/v1/evaluate")
async def evaluate(request: EvalRequest):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.post(f"{EVALUATION_SERVICE_URL}/evaluate", json={
                "dataset_name": request.dataset_name,
                "model_type": request.model_type,
                "top_k": request.top_k,
                "k1": request.k1,
                "b": request.b,
                "max_queries": request.max_queries,
                "use_personalization": request.use_personalization,  # ✅ جديد
                "user_id": request.user_id  # ✅ جديد
            })
            r.raise_for_status()
            return r.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Evaluation service unavailable: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)