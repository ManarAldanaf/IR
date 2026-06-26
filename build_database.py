import sqlite3
import json
import os

DATASETS_DIR = r"C:\Users\User\IR_Project\datasets"
DB_PATH = r"C:\Users\User\IR_Project\datasets\ir_documents.db"

def build_database():
    print("[*] Building SQLite database...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            text TEXT NOT NULL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset ON documents(dataset)")
    conn.commit()

    # ==========================================
    # CORD-19
    # ==========================================
    cord19_path = os.path.join(DATASETS_DIR, "cord19_processed.jsonl")
    if os.path.exists(cord19_path):
        print("[*] Loading CORD-19...")
        count = 0
        with open(cord19_path, 'r', encoding='utf-8') as f:
            batch = []
            for line in f:
                data = json.loads(line)
                text = data.get('text', ' '.join(data.get('tokens', [])))
                batch.append((data['doc_id'], 'cord19', text))
                if len(batch) >= 5000:
                    cursor.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", batch)
                    conn.commit()
                    count += len(batch)
                    print(f"[PROGRESS] CORD-19: {count} documents...")
                    batch = []
            if batch:
                cursor.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", batch)
                conn.commit()
                count += len(batch)
        print(f"[SUCCESS] CORD-19: {count} documents inserted")

    

    # ==========================================
    # CLINICAL
    # ==========================================
    clinical_path = os.path.join(DATASETS_DIR, "clinical_processed.jsonl")
    if os.path.exists(clinical_path):
        print("[*] Loading Clinical...")
        count = 0
        with open(clinical_path, 'r', encoding='utf-8') as f:
            batch = []
            for line in f:
                data = json.loads(line)
                text = data.get('text', ' '.join(data.get('tokens', [])))
                batch.append((data['doc_id'], 'clinical', text))
                if len(batch) >= 5000:
                    cursor.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", batch)
                    conn.commit()
                    count += len(batch)
                    print(f"[PROGRESS] Clinical: {count} documents...")
                    batch = []
            if batch:
                cursor.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", batch)
                conn.commit()
                count += len(batch)
        print(f"[SUCCESS] Clinical: {count} documents inserted")

    conn.close()
    print(f"\n[ALL DONE] Database saved to: {DB_PATH}")

if __name__ == "__main__":
    build_database()