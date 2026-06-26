import os
import json
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Personalization Service (Port 8009)")

HISTORY_DIR = r"C:\Users\User\IR_Project\history"
os.makedirs(HISTORY_DIR, exist_ok=True)

class UserProfileRequest(BaseModel):
    user_id: str

class PersonalizationService:
    def get_user_profile(self, user_id: str) -> Dict:
        """تحليل سجل البحث واستخراج المواضيع المفضلة"""
        history_file = os.path.join(HISTORY_DIR, f"{user_id}.json")
        
        if not os.path.exists(history_file):
            return {"user_id": user_id, "preferred_topics": [], "search_count": 0}
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            word_counts = {}
            for entry in history:
                query = entry.get("query", "")
                for word in query.split():
                    word_lower = word.lower().strip()
                    if len(word_lower) >= 3:
                        word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "user_id": user_id,
                "preferred_topics": [w[0] for w in top_words],
                "search_count": len(history)
            }
        except:
            return {"user_id": user_id, "preferred_topics": [], "search_count": 0}

personalization_service = PersonalizationService()

@app.post("/api/v1/get-user-profile")
async def get_user_profile(request: UserProfileRequest):
    try:
        return personalization_service.get_user_profile(request.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "running", "service": "personalization"}

if __name__ == "__main__":
    print("[*] Starting Personalization Service on port 8009...")
    uvicorn.run(app, host="127.0.0.1", port=8009, reload=False)