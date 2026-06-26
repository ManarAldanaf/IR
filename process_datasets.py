import json
import csv
import os
import spacy
import ir_datasets

PROJECT_DIR = r"C:\Users\User\IR_Project"
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")

os.makedirs(DATASETS_DIR, exist_ok=True)

print("[*] Loading SpaCy...")
nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)

def clean_text_local(text):
    if not text or not isinstance(text, str) or not text.strip():
        return []
    
    doc = nlp(text.lower())
    
    return [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]

# ====================================================
# CORD19
# ====================================================
def process_cord19(csv_path):
    print(f"\n[*] Processing CORD19...")
    
    output = os.path.join(DATASETS_DIR, "cord19_processed.jsonl")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Missing {csv_path}")
        return
    
    count = 0
    with open(csv_path, "r", encoding="utf8") as f, open(output, "w", encoding="utf8") as out:
        reader = csv.DictReader(f)
        
        for row in reader:
            # ✅ تم إزالة شرط limit لتجنب خطأ NameError ومعالجة جميع الوثائق
            
            title = row.get("title", "") or ""
            abstract = row.get("abstract", "") or ""
            
            raw = (title + " " + abstract).strip()
            
            if not raw:
                continue
            
            tokens = clean_text_local(raw)
            
            record = {
                "doc_id": row.get("cord_uid", str(count)),
                "tokens": tokens,
                "text": raw,
                "metadata": {
                    "title": title,
                    "abstract": abstract
                }
            }
            
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            
            if count % 5000 == 0:
                print(f"[CORD19] {count}")
                
    print(f"[DONE] CORD19 → {count}")

# ====================================================
# CLINICAL
# ====================================================
def process_clinical():
    print("\n[*] Processing Clinical...")
    
    dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")
    output = os.path.join(DATASETS_DIR, "clinical_processed.jsonl")
    
    count = 0
    with open(output, "w", encoding="utf8") as out:
        for doc in dataset.docs_iter():
            title = getattr(doc, "title", "") or ""
            summary = getattr(doc, "summary", "") or ""
            detailed = getattr(doc, "detailed_description", "") or ""
            eligibility = getattr(doc, "eligibility", "") or ""
            
            raw = " ".join([title, summary, detailed, eligibility]).strip()
            
            if not raw:
                continue
            
            tokens = clean_text_local(raw)
            
            record = {
                "doc_id": str(doc.doc_id),
                "tokens": tokens,
                "text": raw,
                "metadata": {
                    "title": title,
                    "summary": summary,
                    "detailed_description": detailed,
                    "eligibility": eligibility
                }
            }
            
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            
            if count % 5000 == 0:
                print(f"[CLINICAL] {count}")
                
    print(f"[DONE] Clinical → {count}")

# ====================================================
# RUN
# ====================================================
if __name__ == "__main__":
    cord19_path = r"C:\Users\User\.ir_datasets\cord19\2020-07-16\metadata.csv"
    
    process_cord19(cord19_path)
    process_clinical()
    
    print("\n[ALL DONE]")