import os
import json
import re
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import nltk
from nltk.corpus import wordnet
from spellchecker import SpellChecker

app = FastAPI(title="Query Refinement Service (Port 8004) - النسخة المحسّنة")

# ==========================================
# إعدادات المجلدات
# ==========================================
HISTORY_DIR = r"C:\Users\User\IR_Project\history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# ==========================================
# تهيئة المدقق الإملائي (مرة واحدة فقط)
# ==========================================
spell = SpellChecker()
spell.word_frequency.load_words([
    # إضافة كلمات طبية شائعة لتجنب تصحيحها خطأ
    'covid', 'vaccine', 'diabetes', 'cancer', 'tumor', 'therapy',
    'clinical', 'trial', 'patient', 'drug', 'dose', 'efficacy',
    'randomized', 'placebo', 'chronic', 'acute', 'symptom'
])

# ==========================================
# قائمة الكلمات الممنوعة (Stopwords عامة)
# ==========================================
STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "because", "but", "and", "or", "if", "while",
    "although", "though", "until", "since", "unless", "whereas",
    "whether", "whilst", "run", "get", "make", "do", "have", "be",
    "is", "are", "was", "were", "been", "being", "can", "will",
    "don", "should", "now", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "our", "their",
    "what", "which", "who", "whom"
}

# ==========================================
# Pydantic Models
# ==========================================
class RefineRequest(BaseModel):
    query: str
    dataset_name: str = "cord19"
    user_id: str = "user1"
    use_spell_check: bool = True
    use_synonyms: bool = True
    use_history: bool = True

class ClearHistoryRequest(BaseModel):
    user_id: str = "user1"

# ==========================================
# دوال المساعدة المحسّنة
# ==========================================

def is_valid_word(word: str) -> bool:
    """تتحقق إذا كانت الكلمة صالحة للإضافة (ليست Stopword وطويلة بما يكفي)"""
    if not word or len(word) < 2 or len(word) > 30:
        return False
    if word.lower() in STOPWORDS:
        return False
    # تتأكد إنها تحتوي على حروف فقط (تتجاهل الأرقام والرموز)
    if not re.match(r'^[a-zA-Z]+$', word):
        return False
    return True

def spell_check(query: str) -> str:
    """
    تصحيح إملائي حقيقي باستخدام pyspellchecker.
    يصحح الكلمات الخاطئة ويبقي الصحيحة.
    """
    words = query.split()
    corrected_words = []
    
    for word in words:
        # ننظف الكلمة من علامات الترقيم
        clean_word = re.sub(r'[^\w\s]', '', word)
        if not clean_word:
            corrected_words.append(word)
            continue
            
        # إذا كانت الكلمة صحيحة أو ضمن الكلمات المخصصة، نبقيها
        if clean_word.lower() in spell:
            corrected_words.append(word)
        else:
            # نحاول التصحيح
            correction = spell.correction(clean_word)
            if correction and correction != clean_word.lower():
                # نبدل الكلمة بالتصحيح مع الحفاظ على حالة الحروف الأصلية
                corrected_words.append(correction)
            else:
                # إذا لم نجد تصحيحاً، نبقي الأصل
                corrected_words.append(word)
    
    return " ".join(corrected_words)

def get_synonyms(word: str, max_synonyms: int = 2) -> List[str]:
    """
    استخراج مرادفات حقيقية من WordNet مع ترجيح المصطلحات الطبية.
    """
    word_lower = word.lower()
    synonyms = set()
    
    # قائمة الكلمات الطبية المفضلة (نعطيها أولوية)
    medical_priority = {
        'treatment': ['therapy', 'intervention', 'management'],
        'disease': ['illness', 'condition', 'disorder'],
        'drug': ['medication', 'medicine', 'pharmaceutical'],
        'patient': ['subject', 'participant'],
        'cancer': ['tumor', 'neoplasm', 'carcinoma'],
        'diabetes': ['diabetic', 'hyperglycemia'],
        'heart': ['cardiac', 'cardiovascular'],
        'vaccine': ['immunization', 'vaccination'],
        'virus': ['viral', 'pathogen'],
        'infection': ['infectious', 'contagion'],
        'therapy': ['treatment', 'intervention'],
        'surgery': ['surgical', 'operation'],
        'diagnosis': ['diagnostic', 'detection'],
        'prevention': ['preventive', 'prophylaxis'],
        'symptom': ['sign', 'manifestation'],
        'chronic': ['long-term', 'persistent'],
        'acute': ['severe', 'sudden'],
        'randomized': ['random', 'controlled'],
        'placebo': ['sham', 'control'],
        'efficacy': ['effectiveness', 'potency'],
        'safety': ['security', 'tolerability'],
        'dose': ['dosage', 'amount'],
        'side effect': ['adverse event', 'complication']
    }
    
    # 1. نضيف المرادفات الطبية المفضلة أولاً (من القائمة المخصصة)
    if word_lower in medical_priority:
        for syn in medical_priority[word_lower]:
            if syn != word_lower and is_valid_word(syn):
                synonyms.add(syn)
    
    # 2. بعدها نضيف المرادفات من WordNet (لكن نفلتر الكلمات العامة)
    for synset in wordnet.synsets(word_lower):
        for lemma in synset.lemmas():
            syn = lemma.name().replace('_', ' ').lower()
            # نتأكد إن المرادف ليس نفس الكلمة، وصالح، وليس كلمة عامة ممنوعة
            if syn != word_lower and is_valid_word(syn):
                # نستبعد الكلمات العامة جداً (زي discussion, handling, etc.)
                if syn not in ['discussion', 'handling', 'treatment', 'consideration', 'dealing']:
                    synonyms.add(syn)
    
    # نرجع أول max_synonyms مرادف
    return list(synonyms)[:max_synonyms]

def get_user_history(user_id: str, dataset_name: str, max_words: int = 2) -> List[str]:
    """
    تستخرج كلمات ذات صلة من سجل البحث.
    تحسب الكلمات الأكثر تكراراً في تاريخ المستخدم، وتفلترها حسب الداتاست.
    """
    history_file = os.path.join(HISTORY_DIR, f"{user_id}.json")
    
    if not os.path.exists(history_file):
        return []
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # ✅ فلتر حسب الداتاست (فقط من نفس الداتاست)
        filtered_history = [
            entry for entry in history 
            if entry.get("dataset_name", dataset_name) == dataset_name
        ]
        
        # نحسب تكرار الكلمات في سجل البحث
        word_counts = {}
        for entry in filtered_history:
            query = entry.get("query", "")
            for word in query.split():
                clean_word = re.sub(r'[^\w\s]', '', word).lower()
                if is_valid_word(clean_word):
                    word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
        
        # نرتب الكلمات حسب التكرار ونأخذ الأعلى
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in sorted_words[:max_words]]
    
    except:
        return []

def save_to_history(user_id: str, query: str, dataset_name: str):
    """حفظ الاستعلام في سجل البحث مع اسم الداتاست"""
    history_file = os.path.join(HISTORY_DIR, f"{user_id}.json")
    
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []
    
    history.append({
        "query": query,
        "dataset_name": dataset_name,  # ✅ حفظ اسم الداتاست
        "timestamp": __import__('datetime').datetime.now().isoformat()
    })
    
    # الاحتفاظ بآخر 50 استعلام فقط
    history = history[-50:]
    
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

# ==========================================
# الـ Endpoints
# ==========================================

@app.get("/health")
async def health():
    return {"status": "ok", "service": "query_refinement_enhanced"}

@app.post("/api/v1/refine-query")
async def refine_query(request: RefineRequest):
    try:
        original_query = request.query.strip()
        refined_query = original_query
        steps = []
        
        # 1. التصحيح الإملائي (حقيقي الآن)
        if request.use_spell_check:
            corrected = spell_check(refined_query)
            if corrected != refined_query:
                steps.append({
                    "step": "spell_check",
                    "before": refined_query,
                    "after": corrected
                })
                refined_query = corrected
        
        # 2. إضافة كلمات من سجل البحث (كلمات ذات صلة، بدون تكرار)
        if request.use_history:
            history_words = get_user_history(request.user_id, request.dataset_name, max_words=2)  # ✅ تمرير dataset_name
            if history_words:
                # نأخذ الكلمات الموجودة بالسجل والغير موجودة بالاستعلام الحالي
                current_words = set(refined_query.lower().split())
                new_words = [w for w in history_words if w not in current_words and is_valid_word(w)]
                
                if new_words:
                    before = refined_query
                    refined_query = refined_query + " " + " ".join(new_words)
                    steps.append({
                        "step": "history_weighting",
                        "before": before,
                        "after": refined_query
                    })
        
        # 3. إضافة المرادفات (من WordNet، وليس قاموساً ثابتاً)
        if request.use_synonyms:
            words = refined_query.split()
            new_synonyms = []
            
            for word in words:
                clean_word = re.sub(r'[^\w\s]', '', word).lower()
                if is_valid_word(clean_word):
                    # نأخذ مرادف واحد كحد أقصى لكل كلمة
                    synonyms = get_synonyms(clean_word, max_synonyms=1)
                    for syn in synonyms:
                        # نتأكد إن المرادف غير موجود مسبقاً
                        if syn not in [w.lower() for w in words + new_synonyms]:
                            new_synonyms.append(syn)
            
            # نضيف فقط أول 3 مرادفات جديدة لتجنب تضخيم الاستعلام
            new_synonyms = new_synonyms[:3]
            
            if new_synonyms:
                before = refined_query
                refined_query = refined_query + " " + " ".join(new_synonyms)
                steps.append({
                    "step": "synonym_expansion",
                    "before": before,
                    "after": refined_query
                })
        
        # حفظ الاستعلام الأصلي في السجل مع اسم الداتاست
        save_to_history(request.user_id, original_query, request.dataset_name)  # ✅ تمرير dataset_name
        
        return {
            "original_query": original_query,
            "refined_query": refined_query,
            "refinement_steps": steps,
            "dataset_name": request.dataset_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/clear-history")
async def clear_history(request: ClearHistoryRequest):
    try:
        history_file = os.path.join(HISTORY_DIR, f"{request.user_id}.json")
        if os.path.exists(history_file):
            os.remove(history_file)
        return {"status": "success", "message": "History cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/history/{user_id}")
async def get_history(user_id: str):
    history_file = os.path.join(HISTORY_DIR, f"{user_id}.json")
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

# ==========================================
# تشغيل الخدمة
# ==========================================
if __name__ == "__main__":
    print("[*] Starting Query Refinement Service (Enhanced) on port 8004...")
    uvicorn.run(app, host="127.0.0.1", port=8004, reload=False)