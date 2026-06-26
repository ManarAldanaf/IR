 IR Project 2026 – توثيق بنية الكود (Code Architecture)

هذا المشروع هو نظام استرجاع معلومات (IR) متكامل مبني على مبدأ **SOA (Service-Oriented Architecture)**.  
الهدف من هذا الملف هو توضيح **لبنة الكود** بشكل احترافي: أي ملف مسؤول عن ماذا، وكيف تمر البيانات بين المكونات.

---


 هيكلية المشروع العامة (Project Tree)

C:\Users\User\IR_Project\
│
├── 📁 services/                           # 🔹 كل الخدمات المستقلة (SOA)
│   ├── preprocessing_service.py           # خدمة المعالجة المسبقة (Port 8001)
│   ├── indexing_service.py                # خدمة بناء الفهارس (Port 8002)
│   ├── retrieval_service.py               # خدمة البحث والاسترجاع (Port 8003)
│   ├── query_refinement_service.py        # خدمة تحسين الاستعلامات (Port 8004)
│   ├── evaluation_service.py              # خدمة التقييم والمقاييس (Port 8005)
│   └── document_store_service.py          # خدمة تخزين النصوص في SQLite (Port 8006)
│
├──  main.py                             # API Gateway (البوابة الرئيسية - Port 8000)
├── ui_app.py                           # واجهة المستخدم (Streamlit - بدون Port ثابت)
│
├──  process_datasets.py                 # سكريبت تحميل ومعالجة الداتاست (مرة واحدة)
├── build_database.py                  # سكريبت بناء قاعدة بيانات SQLite (مرة واحدة)
├
│
├── 📂 datasets/                           # (مجلد البيانات) - يتم إنشاؤه تلقائياً
│   ├── cord19_processed.jsonl
│   ├── clinical_processed.jsonl
│   └── ir_documents.db                    # قاعدة البيانات النهائية
│
├── 📂 models/                             # (مجلد الفهارس) - يتم إنشاؤه تلقائياً
│   ├── cord19_bm25.pkl
│   ├── cord19_tfidf.pkl
│   ├── cord19_embeddings.pkl
│   ├── clinical_bm25.pkl
│   ├── clinical_tfidf.pkl
│   └── clinical_embeddings.pkl
│
└──                           