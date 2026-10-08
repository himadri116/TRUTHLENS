"""
TruthLens: Fake News Detection and Credibility Analysis using NLP, RAG, and Generative AI
Interactive Streamlit Application
"""

import os
import sys
import json
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="TruthLens - Fake News & Credibility Analysis",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd

from src.preprocessor import full_nlp_pipeline
from src.bert_classifier import BertNewsClassifier
from src.rag_retriever import RAGRetriever
from src.genai_explainer import GenerativeCredibilityAnalyzer
from src.explainability import TokenExplainer, LimeExplainer, ShapExplainer, XAIComparator
from src.knowledge_base import initialize_knowledge_base, add_fact_check

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .badge-person {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-org {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-gpe {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-date {
        background-color: #f3e8ff;
        color: #6b21a8;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .evidence-box {
        border-left: 4px solid #3b82f6;
        background-color: #f8fafc;
        padding: 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
    .stance-box {
        padding: 14px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .stance-contradicts {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
    }
    .stance-supports {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #4ade80;
    }
    .stance-insufficient {
        background-color: #fef9c3;
        color: #854d0e;
        border: 1px solid #facc15;
    }
</style>
""", unsafe_allow_html=True)


# Cache heavy models to ensure fast responsive app
@st.cache_resource(show_spinner=False)
def load_all_models():
    classifier = BertNewsClassifier()
    retriever = RAGRetriever()
    genai_analyzer = GenerativeCredibilityAnalyzer()
    xai_comparator = XAIComparator(classifier)
    return classifier, retriever, genai_analyzer, xai_comparator


# Initialize models
with st.spinner("Initializing TruthLens models (BERT, RAG Vector Index, XAI Suite with LIME & SHAP)..."):
    classifier, retriever, genai_analyzer, xai_comparator = load_all_models()


# Preloaded sample cases for quick user testing
SAMPLE_CASES = {
    "-- Select a sample news article --": "",
    "🔴 [Fake News] Jennifer Aniston Pregnant With Brad Pitt's Miracle Twins in Paris": (
        "Shocking reports reveal Jennifer Aniston and Brad Pitt have secretly remarried in an intimate "
        "candlelit ceremony in Paris, and Aniston is now expecting miracle twin girls at age 50! "
        "Sources allege the former spouses reconciled during an exotic getaway to France and have spent millions "
        "preparing a nursery at Pitt's French estate Chateau Miraval."
    ),
    "🟢 [Real News] Jennifer Aniston & Justin Theroux Announce Amicable Separation": (
        "Jennifer Aniston and Justin Theroux have announced they are separating after two and a half years of marriage. "
        "In a joint statement released through Aniston's longtime publicist Stephen Huvane, the couple stated the decision was "
        "mutual and lovingly made at the end of last year. They emphasized their commitment to maintaining their deep friendship."
    ),
    "🔴 [Fake News] Queen Elizabeth Officially Strips Harry & Meghan of Royal Bloodlines": (
        "In an unprecedented royal decree, Queen Elizabeth II has signed emergency legislation officially expelling "
        "Prince Harry and Meghan Markle from the British royal lineage entirely. Insiders claim palace guards "
        "confiscated Harry's passport and barred Meghan from ever setting foot on UK soil again."
    ),
    "🟢 [Real News] Prince Harry & Meghan Markle Step Back as Senior Working Royals": (
        "The Duke and Duchess of Sussex announced their decision to step back as senior members of the British Royal Family. "
        "Prince Harry and Meghan Markle released an official Buckingham Palace communication expressing their desire to become "
        "financially independent while continuing to support Queen Elizabeth II."
    ),
    "🔴 [Fake News] Lady Gaga Moved In With Bradley Cooper After Breakup": (
        "Celebrity tabloids claim Lady Gaga has already moved all her belongings into Bradley Cooper's Manhattan townhouse "
        "following Cooper's breakup with Irina Shayk. An alleged housemaid claims Gaga has redecorated the master suite "
        "and the couple are planning an extravagant secret wedding in Italy before the end of the summer."
    ),
    "🟢 [Real News] Lady Gaga Wins Best Original Song Oscar for Shallow": (
        "Lady Gaga took home her first Academy Award for Best Original Song for Shallow from the musical drama film A Star Is Born. "
        "During her emotional acceptance speech at the 91st Academy Awards, she thanked co-star and director Bradley Cooper, "
        "delivering an inspiring message about hard work and discipline in creative pursuits."
    ),
    "🔴 [Fake News] Keanu Reeves Exposed as 500-Year-Old Immortal Vampire": (
        "Shocking historical evidence and facial recognition algorithms allegedly prove Hollywood icon Keanu Reeves "
        "has lived since the 15th century without aging. Researchers claim to have uncovered historical portraits from 1530 "
        "showing Reeves serving as a European physician, holding millions in antique bullion in Zurich."
    ),
    "🟢 [Real News] Leonardo DiCaprio Donates $5M to Amazon Rainforest Relief": (
        "Leonardo DiCaprio's environmental initiative Earth Alliance committed $5 million in financial assistance to preserve "
        "the Amazon rainforest following devastating wildfires across Brazil and Bolivia. The funds were distributed directly "
        "to indigenous communities and local conservation organizations working on frontline emergency response."
    )
}


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search--v1.png", width=64)
    st.title("TruthLens")
    st.caption("AI-Powered News Credibility & Fact Verification")
    st.divider()

    st.subheader("⚙️ System Status")
    st.success("✅ BERT Classifier: Active (DistilBERT)")
    st.success(f"✅ RAG Vector Store: Active ({len(retriever.documents)} Verified Records)")
    st.success("✅ spaCy NER Pipeline: Active (en_core_web_sm)")

    st.divider()
    st.subheader("🤖 Generative AI Configuration")
    api_key_input = st.text_input(
        "Gemini API Key (Optional)",
        type="password",
        help="Optional. If omitted, TruthLens uses the built-in offline neural reasoning engine!"
    )
    if api_key_input:
        genai_analyzer.set_api_key(api_key_input)
        st.info("⚡ Gemini API Connected!")
    else:
        st.info("ℹ️ Using Built-in Local Reasoning Engine (100% offline & free)")

    st.divider()
    st.caption("TruthLens Project • NLP, RAG, and Generative AI")


# ==========================================
# MAIN INTERFACE TABS
# ==========================================
st.markdown('<div class="main-header">TruthLens: Fake News & Credibility Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Combining fine-tuned BERT classification with dense RAG retrieval, fact-checking knowledge bases, and Generative AI reasoning.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Credibility Analyzer",
    "📊 Project Evolution (Old vs New)",
    "📈 Model Evaluation & GossipCop Benchmark",
    "📚 Verified Knowledge Base"
])


# ==========================================
# TAB 1: LIVE CREDIBILITY ANALYZER
# ==========================================
with tab1:
    col_input, col_preset = st.columns([3, 2])

    with col_preset:
        selected_sample = st.selectbox("⚡ Quick Test with GossipCop Benchmark Samples:", list(SAMPLE_CASES.keys()))

    default_text = SAMPLE_CASES[selected_sample] if selected_sample != "-- Select a sample news article --" else ""

    news_input = st.text_area(
        "Enter News Article or Claim to Verify:",
        value=default_text,
        height=140,
        placeholder="Paste a news headline, social media post, or article text here..."
    )

    analyze_btn = st.button("🚀 Analyze Credibility with TruthLens", type="primary", width="stretch")

    if analyze_btn and news_input.strip():
        with st.spinner("Processing news through NLP preprocessing, BERT classifier, FAISS RAG, and GenAI..."):
            # 1. spaCy NLP Pipeline
            nlp_res = full_nlp_pipeline(news_input)

            # 2. BERT Classification
            bert_res = classifier.predict(news_input)

            # 3. RAG Semantic Retrieval
            entity_keywords = nlp_res["ner"]["people"] + nlp_res["ner"]["organizations"]
            retrieved_docs = retriever.retrieve(news_input, top_k=3, entity_boost=entity_keywords)

            # 4. Generative AI Credibility & Stance Synthesis
            credibility_res = genai_analyzer.analyze_credibility(
                news_text=news_input,
                bert_result=bert_res,
                retrieved_evidence=retrieved_docs,
                entities=nlp_res["ner"]
            )

            # 5. Explainable AI Multi-Method Comparison (LIME, SHAP, Perturbation)
            xai_res = xai_comparator.compare(news_input)

        st.divider()

        # TOP ROW: VERDICT & CREDIBILITY SCORE
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        pred_color = "red" if bert_res["prediction"] == "FAKE" else "green"
        with m_col1:
            st.markdown(f"### BERT Verdict\n# :{pred_color}[{bert_res['prediction']}]")
            st.caption(f"Model Confidence: {bert_res['confidence']*100:.1f}%")

        with m_col2:
            st.metric(
                label="Overall Credibility Score",
                value=f"{credibility_res['credibility_score']}%",
                delta=f"{credibility_res['credibility_rating']}"
            )

        with m_col3:
            st.metric(
                label="Evidence Match Stance",
                value=credibility_res["stance"],
                help="Stance determined by comparing the claim against retrieved fact-checking archives."
            )

        with m_col4:
            st.metric(
                label="Evidence Retrieved",
                value=f"{len(retrieved_docs)} Verified Sources",
                help="Number of relevant journalistic fact-check records retrieved via FAISS vector search."
            )

        st.progress(credibility_res["credibility_score"] / 100.0)

        # STANCE BANNER
        stance = credibility_res["stance"]
        stance_class = "stance-contradicts" if stance == "CONTRADICTS" else ("stance-supports" if stance == "SUPPORTS" else "stance-insufficient")
        st.markdown(
            f'<div class="stance-box {stance_class}">'
            f'📌 <b>Evidence Stance:</b> {credibility_res["stance_label"]}<br>'
            f'<span style="font-weight: 400; font-size: 0.92rem;">{credibility_res["stance_details"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # TWO COLUMN LAYOUT: NLP & GENAI EXPLANATION
        left_col, right_col = st.columns([1, 1])

        with left_col:
            # GENERATIVE AI EXPLANATION
            st.subheader("💡 Generative AI Evidence-Based Explanation")
            st.caption(f"Synthesized by: {credibility_res['reasoning_engine']}")
            st.info(credibility_res["ai_explanation"])

            # NAMED ENTITIES (spaCy NER)
            st.subheader("🏷️ Extracted Entities (spaCy NER)")
            ner_data = nlp_res["ner"]
            if ner_data["total_entities_found"] > 0:
                badges_html = ""
                for p in ner_data["people"]:
                    badges_html += f'<span class="badge-person">👤 {p}</span> '
                for o in ner_data["organizations"]:
                    badges_html += f'<span class="badge-org">🏢 {o}</span> '
                for g in ner_data["locations"]:
                    badges_html += f'<span class="badge-gpe">📍 {g}</span> '
                for d in ner_data["dates"]:
                    badges_html += f'<span class="badge-date">📅 {d}</span> '
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.write("No major named entities identified in the text.")

            with st.expander("🔍 Linguistic Preprocessing Details"):
                st.write("**Cleaned Text:**", nlp_res["cleaned_text"])
                st.write("**Lemmatized Form:**", nlp_res["lemmatized_text"])
                st.write("**Extracted Claim Keywords:**", ", ".join(nlp_res["claim_analysis"]["keywords"]))

        with right_col:
            # RAG RETRIEVED EVIDENCE
            st.subheader("📑 RAG Retrieved Fact-Checking Evidence (FAISS)")
            if retrieved_docs:
                for idx, doc in enumerate(retrieved_docs, 1):
                    v_color = "red" if "FALSE" in doc["verdict"] or "DEBUNKED" in doc["verdict"] else "green"
                    doc_url = doc.get("url", "#")
                    explorer_url = doc.get(
                        "factcheck_explorer_url",
                        f"https://toolbox.google.com/factcheck/explorer/search/text:{doc.get('claim_reviewed', '')}"
                    )
                    st.markdown(
                        f'<div class="evidence-box">'
                        f'<b>#{idx} {doc["title"]}</b><br>'
                        f'<b>Source:</b> {doc["source"]} | <b>Relevance:</b> {doc["relevance_percent"]}%<br>'
                        f'<b>Official Verdict:</b> <span style="color: {v_color}; font-weight: bold;">{doc["verdict"]}</span><br>'
                        f'<p style="margin-top: 6px; font-size: 0.93rem; color: #333; line-height: 1.5;">{doc["fact_check_summary"]}</p>'
                        f'<div style="margin-top: 8px; display: flex; gap: 15px; flex-wrap: wrap;">'
                        f'<a href="{doc_url}" target="_blank" style="font-size: 0.85rem; color: #1d4ed8; font-weight: 600; text-decoration: underline;">🔗 Search on {doc["source"]}</a>'
                        f'<a href="{explorer_url}" target="_blank" style="font-size: 0.85rem; color: #047857; font-weight: 600; text-decoration: underline;">🌐 Google Fact Check Explorer</a>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No matching verified fact-checks found in the local knowledge base with sufficient similarity.")

        # ROW 3: MODEL EXPLAINABILITY (XAI)
        st.divider()
        st.subheader("🔬 Model Explainability (XAI): Multi-Method Feature Attribution")
        st.caption("Benchmark how different Explainable AI algorithms (LIME, SHAP, and Leave-One-Out Perturbation) attribute importance to words in the news article.")

        # XAI Mode Selector
        xai_mode = st.radio(
            "Select XAI Analysis / Visualization Mode:",
            [
                "📊 Multi-Method Comparative Analysis (LIME vs SHAP vs Perturbation)",
                "🍋 LIME (Local Interpretable Model-agnostic Explanations)",
                "🔺 SHAP (SHapley Additive exPlanations - Game Theory)",
                "⚡ Fast Perturbation (Leave-One-Out Empirical Baseline)"
            ],
            horizontal=True
        )

        indiv = xai_res.get("individual_results", {})
        pert_data = indiv.get("Perturbation", {})
        lime_data = indiv.get("LIME", {})
        shap_data = indiv.get("SHAP", {})

        if "Comparative Analysis" in xai_mode:
            # Latency and Agreement Banner
            x_m1, x_m2, x_m3, x_m4 = st.columns(4)
            x_m1.metric("LIME & SHAP Agreement", f"{xai_res.get('lime_shap_agreement_percent', 0)}%", help="Jaccard overlap of top influential words identified by both LIME and SHAP.")
            x_m2.metric("Perturbation Latency", f"{xai_res.get('latency', {}).get('Perturbation', 0)} ms", "Fastest")
            x_m3.metric("LIME Latency", f"{xai_res.get('latency', {}).get('LIME', 0)} ms", "Surrogate Model")
            x_m4.metric("SHAP Latency", f"{xai_res.get('latency', {}).get('SHAP', 0)} ms", "Game Theoretic")

            # Consensus words badge
            consensus = xai_res.get("consensus_words", [])
            if consensus:
                st.markdown(
                    f"**🎯 Consensus Salient Tokens (Agreed across methods):** " +
                    " ".join([f"`{w}`" for w in consensus])
                )

            # Side-by-side heatmaps
            st.markdown("#### Side-by-Side Token Heatmaps")
            h_col1, h_col2, h_col3 = st.columns(3)
            with h_col1:
                st.markdown(pert_data.get("html_visualization", ""), unsafe_allow_html=True)
            with h_col2:
                st.markdown(lime_data.get("html_visualization", ""), unsafe_allow_html=True)
            with h_col3:
                st.markdown(shap_data.get("html_visualization", ""), unsafe_allow_html=True)

            # Comparative multi-bar chart
            st.markdown("#### 📊 Comparative Token Importance Scores")
            comp_table = xai_res.get("comparison_table", [])
            if comp_table:
                df_chart = pd.DataFrame(comp_table)
                df_chart.set_index("Token", inplace=True)
                st.bar_chart(df_chart, height=300)

            # In-depth Methodological Breakdown Table
            st.markdown("#### 📋 Theoretical & Practical Comparison of XAI Methods")
            st.dataframe(pd.DataFrame(xai_res.get("methodology_comparison", [])), width="stretch", hide_index=True)

        elif "LIME" in xai_mode:
            st.markdown(lime_data.get("html_visualization", ""), unsafe_allow_html=True)
            st.caption(f"Execution Latency: {lime_data.get('execution_time_ms', 0)} ms | Local Surrogate Linear Model")
            if lime_data.get("top_influential_words"):
                st.write("**Top Words Identified by LIME:**")
                cols = st.columns(len(lime_data["top_influential_words"]))
                for idx, item in enumerate(lime_data["top_influential_words"]):
                    with cols[idx]:
                        st.metric(label=f"Token: '{item['word']}'", value=f"{item['importance']:.4f}", delta=item["impact"])

        elif "SHAP" in xai_mode:
            st.markdown(shap_data.get("html_visualization", ""), unsafe_allow_html=True)
            st.caption(f"Execution Latency: {shap_data.get('execution_time_ms', 0)} ms | Cooperative Game Theory Shapley Values")
            if shap_data.get("top_influential_words"):
                st.write("**Top Words Identified by SHAP:**")
                cols = st.columns(len(shap_data["top_influential_words"]))
                for idx, item in enumerate(shap_data["top_influential_words"]):
                    with cols[idx]:
                        st.metric(label=f"Token: '{item['word']}'", value=f"{item['importance']:.4f}", delta=item["impact"])

        else:  # Perturbation
            st.markdown(pert_data.get("html_visualization", ""), unsafe_allow_html=True)
            st.caption(f"Execution Latency: {pert_data.get('execution_time_ms', 0)} ms | Leave-One-Out Marginal Perturbation")
            if pert_data.get("top_influential_words"):
                st.write("**Top Words Identified by Perturbation:**")
                cols = st.columns(len(pert_data["top_influential_words"]))
                for idx, item in enumerate(pert_data["top_influential_words"]):
                    with cols[idx]:
                        st.metric(label=f"Token: '{item['word']}'", value=f"{item['importance']:.4f}", delta=item["impact"])


# ==========================================
# TAB 2: PROJECT EVOLUTION (OLD VS NEW)
# ==========================================
with tab2:
    st.header("Project Evolution: Previous Semester vs New Semester")
    st.write(
        "In the previous semester, the system was built as a traditional NLP and BERT classifier. "
        "While it achieved strong classification accuracy, it operated as a closed black box. "
        "The new project transforms the system into **TruthLens**, a comprehensive credibility analysis suite powered by RAG and Generative AI."
    )

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.markdown("### 🏛️ Previous Semester Work (Baseline)")
        st.markdown("""
        - **Primary Task**: Binary sequence classification (Real vs Fake).
        - **Dataset**: GossipCop dataset containing celebrity news and rumors.
        - **NLP Pipeline**: Text normalization, token cleaning, and lemmatization using spaCy.
        - **Core Model**: Pretrained BERT (`distilbert-base-uncased`) fine-tuned for classification.
        - **Named Entity Recognition**: Identified entities (`PERSON`, `ORG`, `GPE`) using spaCy.
        - **Evaluation**: Accuracy, Precision, Recall, Confusion Matrix, and ~0.92 F1-score.
        - **Interface**: Simple Streamlit app returning label and raw confidence score.
        - ⚠️ **Key Limitations Identified**:
          - Black-box prediction: Unable to explain *why* an article was flagged.
          - No external verification: Could not cross-reference assertions against external truth.
          - Prone to hallucinations or stale knowledge on unseen real-world claims.
        """)

    with comp_col2:
        st.markdown("### 🚀 New Semester Work (TruthLens: RAG + GenAI + XAI)")
        st.markdown("""
        - **Claim & Semantic Entity Extraction**: Extracts core factual claims and search terms.
        - **Retrieval-Augmented Generation (RAG)**:
          - Curated knowledge base of verified fact-checks (Snopes, PolitiFact, Reuters, AP).
          - Dense embeddings with `SentenceTransformer` (`all-MiniLM-L6-v2`).
          - FAISS vector database providing sub-second semantic retrieval.
        - **Generative AI Evidence Synthesis**:
          - Analyzes news claim alongside retrieved verified articles.
          - Detects evidentiary stance: **Supports**, **Contradicts**, or **Insufficient Evidence**.
          - Generates evidence-based human-readable explanations.
          - Computes an integrated 0-100% **Credibility Score**.
        - **Multi-Method Model Explainability (XAI)**:
          - Integrates **LIME** (sparse linear surrogates), **SHAP** (Shapley game-theoretic values), and empirical perturbation.
          - Provides side-by-side comparative token attributions, consensus salient words, and latency benchmarking.
        - **Comprehensive Interface**:
          - Multi-view dashboard with side-by-side comparison, metric curves, XAI studio, and knowledge base explorer.
        """)

    st.divider()
    st.subheader("System Architecture Comparison")
    st.code("""
    PREVIOUS SEMESTER FLOW:
    [News Text] ---> [spaCy Preprocessing] ---> [Fine-Tuned BERT] ---> [Prediction (Real/Fake) + Confidence]
    
    NEW SEMESTER (TRUTHLENS) FLOW:
    [News Text] ---> [spaCy Preprocessing & NER] ---> [Fine-Tuned BERT] ---> [Prediction]
                            |                                |                       |
                            |                                |                       v
                            |                                +-----------------> [XAI Suite: LIME + SHAP + Perturbation]
                            v                                v
    [FAISS Vector Store] -> [RAG Semantic Retrieval] ----------> [Generative AI Reasoning Engine]
                                                                     |
                                                                     v
                                              [Evidence Stance + AI Explanation + Credibility Score]
    """, language="text")


# ==========================================
# TAB 3: MODEL EVALUATION & GOSSIPCOP BENCHMARK
# ==========================================
with tab3:
    st.header("BERT Model Evaluation on GossipCop Benchmark")
    st.write("Quantitative performance metrics from fine-tuning BERT on the GossipCop dataset (Previous Semester Foundation).")

    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Accuracy", f"{metrics_data['accuracy']*100:.1f}%")
        m2.metric("Precision", f"{metrics_data['precision']*100:.1f}%")
        m3.metric("Recall", f"{metrics_data['recall']*100:.1f}%")
        m4.metric("F1-Score", f"{metrics_data['f1_score']:.4f}", help="Achieved target benchmark of ~0.92 F1 score")

        st.divider()

        plot_col, table_col = st.columns([1.2, 1])

        with plot_col:
            plot_img_path = "models/confusion_matrix.png"
            if os.path.exists(plot_img_path):
                st.image(plot_img_path, caption="Confusion Matrix and Training Loss Curve", width="stretch")

        with table_col:
            st.subheader("Per-Class Classification Report")
            per_class = metrics_data.get("per_class_metrics", {})
            rows = []
            for c_name, scores in per_class.items():
                rows.append({
                    "Class": c_name,
                    "Precision": f"{scores.get('precision', 0.0):.4f}",
                    "Recall": f"{scores.get('recall', 0.0):.4f}",
                    "F1-Score": f"{scores.get('f1_score', 0.0):.4f}"
                })
            st.dataframe(rows, width="stretch")

            st.markdown("""
            **Key Evaluation Observations:**
            - The fine-tuned Transformer model successfully separates sensational tabloid rumors from verified news reports.
            - F1-Score of **~0.92** confirms balanced precision and recall across both Real and Fake classes.
            - Low false positive and false negative rates validate the efficacy of spaCy preprocessing and lemmatization.
            """)
    else:
        st.warning("Metrics file not found. Please run `python src/train_evaluate.py` to regenerate metrics.")


# ==========================================
# TAB 4: VERIFIED KNOWLEDGE BASE EXPLORER
# ==========================================
with tab4:
    st.header("Verified Fact-Checking Knowledge Base")
    st.write("Explore and search the verified fact-checking records indexed in the FAISS vector database.")

    all_docs = initialize_knowledge_base()

    search_kw = st.text_input("Search Fact-Check Database by keyword:", placeholder="e.g. Brad Pitt, Lady Gaga, Harry, Vatican...")

    filtered_docs = [
        d for d in all_docs
        if search_kw.lower() in d["title"].lower() or search_kw.lower() in d["claim_reviewed"].lower() or search_kw.lower() in d["fact_check_summary"].lower()
    ] if search_kw else all_docs

    st.write(f"Showing **{len(filtered_docs)}** of **{len(all_docs)}** records:")

    for d in filtered_docs:
        with st.expander(f"📌 [{d['verdict']}] {d['title']} ({d['source']})"):
            st.write("**Claim Reviewed:**", d["claim_reviewed"])
            st.write("**Official Verdict:**", d["verdict"])
            st.write("**Source:**", d["source"])
            st.write("**Evidence Summary:**", d["fact_check_summary"])
            tab4_url = d.get("url", "#")
            tab4_explorer = d.get("factcheck_explorer_url", f"https://toolbox.google.com/factcheck/explorer/search/text:{d.get('claim_reviewed', '')}")
            st.markdown(
                f'<div style="margin-top: 6px; display: flex; gap: 15px;">'
                f'<a href="{tab4_url}" target="_blank" style="color: #1d4ed8; font-weight: 600;">🔗 Search on {d["source"]}</a>'
                f'<a href="{tab4_explorer}" target="_blank" style="color: #047857; font-weight: 600;">🌐 Google Fact Check Explorer</a>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.divider()
    st.subheader("➕ Add New Fact-Check Record to Vector Index")
    with st.form("add_fact_check_form"):
        f_title = st.text_input("Article Title:")
        f_claim = st.text_input("Claim Investigated:")
        f_verdict = st.selectbox("Verdict:", ["FALSE / DEBUNKED", "TRUE / VERIFIED", "MISLEADING / UNPROVEN"])
        f_source = st.text_input("Publisher / Source (e.g. Snopes, Reuters, AP):")
        f_category = st.text_input("Category:", value="Celebrity & Entertainment")
        f_summary = st.text_area("Detailed Evidence / Journalistic Summary:")
        f_url = st.text_input("Reference URL:")

        submitted = st.form_submit_button("Index into Knowledge Base & FAISS")
        if submitted and f_title and f_summary:
            new_record = {
                "id": f"KB-CUSTOM-{len(all_docs)+1:03d}",
                "title": f_title,
                "claim_reviewed": f_claim,
                "verdict": f_verdict,
                "fact_check_summary": f_summary,
                "source": f_source,
                "url": f_url or "https://factcheck.org",
                "category": f_category
            }
            add_fact_check(new_record)
            # Rebuild index
            retriever.build_index(force_rebuild=True)
            st.success(f"Successfully added '{f_title}' and rebuilt FAISS vector index!")
            st.rerun()
