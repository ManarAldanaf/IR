import streamlit as st
import requests
import pandas as pd
import re

def clean_html_tags(text: str) -> str:
    """يشيل أي HTML tags متبقية من النص الأصلي قبل العرض"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'<[^>]+$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.set_page_config(
    page_title="IR Search Engine 2026",
    page_icon="🔍",
    layout="wide"
)

GATEWAY_URL = "http://127.0.0.1:8000"
DOC_STORE_URL = "http://127.0.0.1:8006"

st.markdown("""
<style>
.main-title { 
    font-size: 2.5rem; 
    font-weight: 700; 
    color: #1a1a2e; 
    text-align: center; 
    margin-bottom: 0.5rem; 
}
.subtitle { 
    text-align: center; 
    color: #666; 
    margin-bottom: 2rem; 
}
.result-card { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    border-left: 4px solid #ffd700; 
    padding: 1.5rem; 
    margin: 1rem 0; 
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    color: #ffffff;
}
.score-badge { 
    background: #ffd700; 
    color: #1a1a2e; 
    padding: 4px 12px; 
    border-radius: 12px; 
    font-size: 0.85rem;
    font-weight: bold;
    display: inline-block;
}
.rank-badge { 
    background: #ff6b6b; 
    color: white; 
    padding: 4px 12px; 
    border-radius: 12px; 
    font-size: 0.85rem;
    font-weight: bold;
    display: inline-block;
}
.doc-id-box {
    background: rgba(255,255,255,0.2);
    color: #ffffff;
    padding: 8px 12px;
    border-radius: 6px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    margin: 0.5rem 0;
    border: 1px solid rgba(255,255,255,0.3);
}
.refined-query { 
    background: #e8f4e8; 
    border-left: 4px solid #28a745; 
    padding: 1rem; 
    border-radius: 4px; 
    margin: 0.5rem 0;
    color: #1a1a2e;
}
.doc-text { 
    color: #f0f0f0; 
    font-size: 0.9rem; 
    margin-top: 0.5rem; 
    border-top: 1px solid rgba(255,255,255,0.3); 
    padding-top: 0.5rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔍 Information Retrieval System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">نظام استرجاع المعلومات — CORD-19 & Clinical Trials</div>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("⚙️ إعدادات البحث")
    
    dataset = st.selectbox(
        "📂 اختر مجموعة البيانات",
        ["cord19", "clinical"],
        format_func=lambda x: {
            "cord19": "📚 CORD-19 (أبحاث COVID-19)",
            "clinical": "🏥 Clinical Trials (تجارب سريرية)"
        }[x]
    )

    st.subheader("🤖 نموذج التمثيل")
    mode = st.radio(
        "اختر وضع التنفيذ:",
        ["أساسي فقط", "أساسي + إضافي"],
        help="أساسي: BM25, TF-IDF, Embedding, Hybrid | إضافي: يشمل Query Refinement + Personalization"
    )

    model_type = st.selectbox(
        "النموذج",
        ["bm25", "tfidf", "embedding", "hybrid_serial", "hybrid_parallel"],
        format_func=lambda x: {
            "bm25": "BM25 (احتمالي)",
            "tfidf": "TF-IDF (VSM)",
            "embedding": "Embedding (دلالي)",
            "hybrid_serial": "🔗 Hybrid Serial (تسلسلي)",
            "hybrid_parallel": "⚡ Hybrid Parallel (متوازي/تفرعي)"
        }[x]
    )

    if model_type == "hybrid_serial":
        st.info("التسلسلي: TF-IDF يرشّح (سريع) ← Embedding يعيد الترتيب")
    elif model_type == "hybrid_parallel":
        st.info("المتوازي: TF-IDF + Embedding بالتوازي ← RRF يدمج النتائج")

    if model_type == "bm25":
        st.subheader("🔧 معاملات BM25")
        k1 = st.slider("k1 (تشبع المصطلح)", 0.5, 3.0, 1.5, 0.1,
                       help="k1 أعلى = وزن أكبر للتكرار")
        b = st.slider("b (تطبيع الطول)", 0.0, 1.0, 0.75, 0.05,
                      help="b=1: تطبيع كامل | b=0: بدون تطبيع")
        st.caption(f"المعاملات الحالية: k1={k1}, b={b}")
    else:
        k1 = 1.5
        b = 0.75

    st.subheader("✨ Query Refinement")
    use_refinement = st.checkbox(
        "تفعيل تحسين الاستعلام",
        value=False,
        help="تحسين الاستعلام بالتصحيح الإملائي والمرادفات وسجل البحث"
    )

    if use_refinement:
        spell_check = st.checkbox("تصحيح إملائي", value=True)
        synonyms = st.checkbox("إضافة مرادفات", value=True)
        history_w = st.checkbox("تثقيل بسجل البحث", value=True)
    else:
        spell_check = synonyms = history_w = False

    # ✅ Personalization (الطلب الإضافي - البند 16)
    use_personalization = False
    if mode == "أساسي + إضافي":
        st.subheader("🎯 التخصيص الشخصي (Personalization)")
        use_personalization = st.checkbox(
            "تفعيل التخصيص الشخصي",
            value=True,
            help="النظام يتعلم من تاريخ بحثك ويخصص النتائج لك"
        )

    st.subheader("📊 النتائج")
    top_k = st.slider("عدد النتائج", 5, 100, 10)
    show_text = st.checkbox("عرض نص الوثيقة", value=True)
    text_preview = st.slider("طول المعاينة (حرف)", 100, 500, 200) if show_text else 200
    user_id = st.text_input("معرف المستخدم", value="user1")

# ==========================================
# البحث
# ==========================================
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "🔎 أدخل استعلامك",
        placeholder="مثال: covid vaccine effectiveness | clinical trial diabetes",
        label_visibility="collapsed"
    )
with col2:
    search_btn = st.button(
        "بحث 🔍",
        use_container_width=True,
        type="primary"
    )

if search_btn and query:
    with st.spinner("جاري البحث..."):

        refined_query = query
        refinement_steps = []

        # Query refinement
        if use_refinement:
            try:
                r = requests.post(
                    f"{GATEWAY_URL}/api/v1/refine-query",
                    json={
                        "query": query,
                        "dataset_name": dataset,
                        "user_id": user_id,
                        "use_spell_check": spell_check,
                        "use_synonyms": synonyms,
                        "use_history": history_w
                    },
                    timeout=300
                )

                if r.status_code == 200:
                    data = r.json()
                    refined_query = data.get("refined_query", query)
                    refinement_steps = data.get("refinement_steps", [])
            except:
                pass

        try:
            r = requests.post(
                f"{GATEWAY_URL}/api/v1/search",
                json={
                    "dataset_name": dataset,
                    "query": refined_query,
                    "model_type": model_type,
                    "top_k": top_k,
                    "k1": k1,
                    "b": b,
                    "use_refinement": False,
                    "use_personalization": use_personalization,
                    "user_id": user_id,
                    "fetch_text": show_text
                },
                timeout=300
            )

            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                tokens = data.get("query_tokens", [])

                st.subheader("📋 معلومات الاستعلام")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("الداتاسيت", dataset.upper())
                with c2:
                    labels = {
                        "bm25": "BM25", "tfidf": "TF-IDF", "embedding": "Embedding",
                        "hybrid_serial": "Hybrid Serial", "hybrid_parallel": "Hybrid Parallel"
                    }
                    st.metric("النموذج", labels[model_type])
                with c3:
                    st.metric("عدد النتائج", len(results))
                with c4:
                    st.metric("Query Refinement", "مفعّل ✅" if use_refinement else "غير مفعّل")

                if model_type == "bm25":
                    st.caption(f"🔧 معاملات BM25: k1={k1}, b={b}")

                # ✅ عرض User Profile إذا كان Personalization مفعل
                if use_personalization and "user_profile" in data:
                    profile = data["user_profile"]
                    st.info(f"📊 **User Profile:** {profile['search_count']} عمليات بحث سابقة")
                    if profile["preferred_topics"]:
                        st.write(f"🎯 **المواضيع المفضلة:** {', '.join(profile['preferred_topics'])}")

                if refinement_steps:
                    st.subheader("✨ خطوات تحسين الاستعلام")
                    for step in refinement_steps:
                        st.markdown(
                            f"""
<div class="refined-query">
    <b>{step['step']}</b><br>
قبل: <code>{step['before']}</code><br>
بعد: <code>{step['after']}</code>
</div>
""",
                            unsafe_allow_html=True
                        )

                if tokens:
                    st.caption("🔤 كلمات الاستعلام: " + " | ".join(tokens))

                st.divider()
                st.subheader(f"📄 النتائج ({len(results)})")

                if not results:
                    st.warning("لم يتم العثور على نتائج.")
                else:
                    for res in results:
                        doc_id = res["doc_id"]
                        text = clean_html_tags(res.get("text", ""))
                        preview = text[:text_preview] + "..." if len(text) > text_preview else text
                        text_html = f'<div class="doc-text">{preview}</div>' if preview else ""
                        
                        st.markdown(
                            f"""
<div class="result-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <span class="rank-badge">#{res['rank']}</span>
        <span class="score-badge">Score: {res['score']:.4f}</span>
    </div>
    <div class="doc-id-box">
📄 <b>Document ID:</b> {doc_id}
    </div>
{text_html}
</div>
""",
                            unsafe_allow_html=True
                        )
            else:
                st.error(f"خطأ: {r.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ تعذر الاتصال بالخدمات")
        except Exception as e:
            st.error(f"خطأ: {str(e)}")
elif search_btn:
    st.warning("⚠️ الرجاء إدخال استعلام")

# ==========================================
# التقييم (مع عرض الوثائق المسترجعة)
# ==========================================
st.divider()
with st.expander("📊 تقييم النظام (Evaluation)", expanded=False):
    st.subheader("حساب مقاييس التقييم")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        eval_dataset = st.selectbox(
            "الداتاسيت",
            ["cord19", "clinical"],
            key="eval_ds"
        )

    with col2:
        eval_model = st.selectbox(
            "النموذج",
            ["bm25", "tfidf", "embedding", "hybrid_serial", "hybrid_parallel"],
            format_func=lambda x: {
                "bm25": "BM25",
                "tfidf": "TF-IDF",
                "embedding": "Embedding",
                "hybrid_serial": "Hybrid Serial",
                "hybrid_parallel": "Hybrid Parallel"
            }[x],
            key="eval_model"
        )

    with col3:
        max_queries = st.slider(
            "عدد الاستعلامات (0 = الكل)",
            0, 100, 0,
            help="0 = استخدام كل الكويريات في qrels",
            key="eval_q"
        )

    with col4:
        top_k_eval = st.slider(
            "عدد النتائج للتقييم (top_k)",
            10, 1000, 100,
            key="top_k_eval"
        )

    # ✅ Personalization في التقييم
    use_personalization_eval = st.checkbox(
        "تفعيل Personalization في التقييم",
        value=False,
        help="تقييم مع Personalization (حسب سجل البحث لنفس الداتاست)"
    )

    dataset_info = {
        "cord19": {"queries": 50, "qrels": 69318, "url": "https://ir-datasets.com/cord19/trec-covid"},
        "clinical": {"queries": 75, "qrels": 35832, "url": "https://ir-datasets.com/clinicaltrials/2021/trec-ct-2021"}
    }

    info = dataset_info[eval_dataset]
    st.info(f"""
📊 معلومات الداتاسيت:
عدد الكويريات: {info['queries']}
عدد الـ qrels: {info['qrels']}
رابط الداتاسيت: {info['url']}
""")

    if st.button("تشغيل التقييم ⚙️", type="secondary"):
        with st.spinner(f"جاري التقييم... (قد يستغرق 5-30 دقيقة)"):
            try:
                r = requests.post(
                    f"{GATEWAY_URL}/api/v1/evaluate",
                    json={
                        "dataset_name": eval_dataset,
                        "model_type": eval_model,
                        "top_k": top_k_eval,
                        "max_queries": max_queries,
                        "use_personalization": use_personalization_eval,  # ✅ جديد
                        "user_id": user_id  # ✅ جديد
                    },
                    timeout=2700
                )

                if r.status_code == 200:
                    data = r.json()
                    metrics = data.get("metrics", {})
                    num_q = data.get("num_queries_evaluated", 0)
                    total_q = data.get("total_queries_in_dataset", 0)
                    st.success(f"✅ تم تقييم {num_q} استعلام (من أصل {total_q} في الداتاسيت)")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAP", f"{metrics.get('MAP', 0):.4f}")
                    with col2:
                        st.metric("Recall", f"{metrics.get('Recall', 0):.4f}")
                    with col3:
                        st.metric("Precision@10", f"{metrics.get('Precision@10', 0):.4f}")
                    with col4:
                        st.metric("nDCG", f"{metrics.get('nDCG', 0):.4f}")

                    per_query = data.get("per_query_results", [])
                    if per_query:
                        st.subheader("📋 تفاصيل كل استعلام")

                        df_data = []
                        for pq in per_query:
                            df_data.append({
                                "Query ID": pq.get("query_id", ""),
                                "AP": pq.get("AP", 0),
                                "Recall": pq.get("Recall", 0),
                                "P@10": pq.get("Precision@10", 0),
                                "nDCG": pq.get("nDCG", 0),
                                "مسترجعة": pq.get("retrieved_count", 0),
                                "ذات صلة": pq.get("relevant_count", 0),
                                "الوثائق المسترجعة (أول 10)": ", ".join(pq.get("retrieved_docs", [])[:10]),
                                "الوثائق ذات الصلة (أول 10)": ", ".join(pq.get("relevant_docs", [])[:10])
                            })

                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True, height=600)

                        import json
                        from datetime import datetime

                        filename = f"evaluation_{eval_dataset}_{eval_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        json_data = json.dumps(data, indent=2, ensure_ascii=False)

                        st.download_button(
                            label="📥 تحميل النتائج الكاملة (JSON)",
                            data=json_data,
                            file_name=filename,
                            mime="application/json"
                        )

                        st.subheader("🔍 عرض تفصيلي لكل استعلام")
                        for pq in per_query:
                            with st.expander(f"Query {pq['query_id']}"):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("AP", f"{pq.get('AP', 0):.4f}")
                                with col2:
                                    st.metric("Recall", f"{pq.get('Recall', 0):.4f}")
                                with col3:
                                    st.metric("P@10", f"{pq.get('Precision@10', 0):.4f}")
                                with col4:
                                    st.metric("nDCG", f"{pq.get('nDCG', 0):.4f}")

                                st.write(f"**عدد الوثائق المسترجعة:** {pq.get('retrieved_count', 0)}")
                                st.write(f"**عدد الوثائق ذات الصلة:** {pq.get('relevant_count', 0)}")

                                st.write("**الوثائق المسترجعة (أول 30):**")
                                st.code(", ".join(pq.get("retrieved_docs", [])[:30]))

                                st.write("**الوثائق ذات الصلة من qrels (أول 30):**")
                                st.code(", ".join(pq.get("relevant_docs", [])[:30]))
                else:
                    st.error(f"خطأ: {r.json().get('detail', '')}")
            except Exception as e:
                st.error(f"خطأ: {str(e)}")

st.divider()
st.caption("IR Project 2026 — Damascus University | Faculty of Information Technology Engineering")