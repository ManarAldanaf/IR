import sqlite3
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(title="Document Store Service (Port 8006)")
DB_PATH = r"C:\Users\User\IR_Project\datasets\ir_documents.db"

# ✅ كاش للوثائق الأكثر طلباً
DOC_CACHE: Dict[str, Dict[str, str]] = {}

class DocRequest(BaseModel):
    dataset_name: str
    doc_ids: List[str]

@app.post("/get-documents")
async def get_documents(request: DocRequest):
    try:
        if not os.path.exists(DB_PATH):
            raise HTTPException(status_code=404, detail=f"Database not found: {DB_PATH}")
        
        # التحقق من الكاش أولاً
        cache_key = request.dataset_name
        if cache_key not in DOC_CACHE:
            DOC_CACHE[cache_key] = {}
        
        # الوثائق المطلوبة التي ليست في الكاش
        missing_ids = [doc_id for doc_id in request.doc_ids if doc_id not in DOC_CACHE[cache_key]]
        
        if missing_ids:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            placeholders = ", ".join(["?" for _ in missing_ids])
            query = f"SELECT doc_id, text FROM documents WHERE dataset = ? AND doc_id IN ({placeholders})"
            
            cursor.execute(query, [request.dataset_name] + missing_ids)
            rows = cursor.fetchall()
            
            # إضافة للكاش
            for doc_id, text in rows:
                DOC_CACHE[cache_key][doc_id] = text
            
            conn.close()
        
        # إرجاع النتائج من الكاش
        result_docs = {doc_id: DOC_CACHE[cache_key].get(doc_id, "") for doc_id in request.doc_ids}
        
        return {"documents": result_docs}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    exists = os.path.exists(DB_PATH)
    cache_size = sum(len(v) for v in DOC_CACHE.values())
    return {"status": "up" if exists else "db_not_found", "cache_size": cache_size}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8006, reload=False)