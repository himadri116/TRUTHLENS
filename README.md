# TruthLens: Fake News Detection and Credibility Analysis using NLP, RAG, and Generative AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Transformers: HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)
[![Vector DB: FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

---

## 📌 Project Overview & Problem Statement

The rapid spread of information across online platforms and social media networks has led to an exponential rise in misinformation. Traditional fake news detection systems classify content as real or fake using supervised models, but they operate as **opaque black boxes**:
1. They lack **explainability** (users do not know *why* a decision was made).
2. They do not cross-reference **external evidence** to verify factual claims.
3. They cannot provide actionable, human-interpretable reasoning when evaluating unseen news claims.

**TruthLens** bridges this gap. It builds upon previous transformer-based classification by integrating **Retrieval-Augmented Generation (RAG)**, a **verified fact-checking knowledge base**, and **Generative AI reasoning** to produce evidence-backed credibility assessments.

---

## 🏛️ Previous Semester Work Done (Foundation)

### Objective
To develop an NLP-based fake news classification pipeline using fine-tuned Transformers on the **GossipCop** benchmark dataset.

### Methodology & Implemented Components
- **Dataset**: Curated GossipCop dataset (celebrity news and rumor claims) partitioned into stratified Train (70%), Validation (15%), and Test (15%) splits.
- **NLP Preprocessing**: Text normalization, URL/HTML tag stripping, tokenization, stop-word removal, and lemmatization using **spaCy** (`en_core_web_sm`).
- **Transformer Fine-Tuning**: Fine-tuned BERT / DistilBERT (`distilbert-base-uncased`) with a sequence classification head, AdamW optimizer, warmup linear learning rate scheduler, and cross-entropy loss.
- **Named Entity Recognition (NER)**: Integrated spaCy entity extraction to identify `PERSON`, `ORGANIZATION`, `GPE` (locations), and `DATE`.
- **Evaluation**: The model achieved an **F1-score of ~0.92**, with high precision (~0.923) and recall (~0.915) on the test split.
- **Streamlit Baseline**: Simple web interface taking user text and outputting a Real/Fake label and confidence score.

### Identified Limitations
The baseline model could only output a label and a statistical probability score. It could not explain its internal reasoning, cite primary sources, or corroborate whether a claim was debunked by journalistic fact-checkers.

---

## 🚀 New Semester Work (TruthLens Enhancements)

TruthLens expands the previous semester system into an evidence-driven credibility analysis platform:

1. **Claim and Semantic Entity Analysis**:
   - Advanced linguistic parsing extracting core claim statements, noun chunks, and search keywords alongside Named Entities (`PERSON`, `ORG`, `GPE`, `DATE`).
2. **Retrieval-Augmented Generation (RAG) with FAISS**:
   - Curated knowledge base of verified fact-checks and official releases (Snopes, PolitiFact, Reuters Fact Check, AP News).
   - Dense semantic vector embeddings powered by `SentenceTransformer` (`all-MiniLM-L6-v2`).
   - Indexed using **FAISS** (Facebook AI Similarity Search) with cosine similarity inner-product search.
3. **Generative AI Credibility & Stance Analysis**:
   - Stance detection: Analyzes whether external evidence **Supports the claim**, **Contradicts the claim**, or **Does not provide enough evidence**.
   - Generates an evidence-backed narrative explanation.
   - Dual-Mode reasoning: Works out-of-the-box using the built-in offline neural reasoning engine, with optional support for Google Gemini API (`gemini-1.5-flash`).
   - Computes an integrated **Credibility Score (0 - 100%)** combining BERT confidence with evidentiary stance.
4. **Multi-Method Explainable AI (XAI) Suite**:
   - **LIME (Local Interpretable Model-agnostic Explanations)**: Builds sparse linear surrogate models around sampled perturbations.
   - **SHAP (SHapley Additive exPlanations)**: Uses cooperative game theory to assign fair marginal contributions to each word.
   - **Leave-One-Out Empirical Perturbation**: Fast baseline measuring probability shifts.
   - **XAI Benchmark Studio**: Side-by-side comparative heatmaps, multi-bar token charts, agreement metrics, and latency profiling.
5. **Interactive Multi-Tab Dashboard**:
   - Live Credibility Analyzer with preloaded benchmark test cases.
   - Project Evolution side-by-side comparison tab.
   - Model Evaluation tab displaying confusion matrices and training loss curves.
   - Verified Knowledge Base explorer with dynamic fact-check submission.

---

## 🏗️ Architecture & Data Flow

```
+-------------------------------------------------------------------------------+
|                               User News Input                                 |
+-------------------------------------------------------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
      [spaCy NLP Preprocessing]                [Fine-Tuned BERT Classifier]
      - Text Cleaning & Normalization          - Binary Sequence Classification
      - spaCy Lemmatization                   - Softmax Class Probabilities
      - NER (PERSON, ORG, GPE, DATE)           - Token Attribution (XAI Heatmap)
      - Main Claim & Keyword Extraction                    |
                   |                                       |
                   v                                       |
      [SentenceTransformer Embedding]                      |
                   |                                       |
                   v                                       |
      [FAISS Vector Database Index]                        |
      - Semantic Cosine Similarity Search                  |
      - Top-k Verified Fact-Check Retrieval               |
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
                       [Generative AI Reasoning Engine]
                       - Evidentiary Stance Detection
                       - Evidence-Backed Fact Explanation
                       - Calibrated Credibility Score (0-100%)
                                       |
                                       v
                       [Streamlit Interactive Dashboard]
```

---

## 📂 Project Structure

```
TruthLens/
├── app.py                      # Streamlit interactive web application
├── requirements.txt            # Project dependencies
├── test_truthlens.py           # Automated unit and integration test suite
├── README.md                   # Complete documentation
├── data/
│   ├── gossipcop_full.csv      # Complete GossipCop dataset
│   ├── gossipcop_train.csv     # Training set split (70%)
│   ├── gossipcop_val.csv       # Validation set split (15%)
│   └── gossipcop_test.csv      # Test set split (15%)
├── knowledge_base/
│   ├── fact_check_database.json # Verified fact-checking knowledge records
│   ├── faiss_index.bin         # Serialized FAISS vector index
│   └── faiss_metadata.json     # Metadata catalog for retrieved documents
├── models/
│   ├── bert_fake_news_detector/ # Fine-tuned BERT model weights & tokenizer
│   ├── metrics.json            # Quantitative test metrics (F1 ~0.92, Precision, Recall)
│   └── confusion_matrix.png    # Confusion matrix and training loss plots
└── src/
    ├── __init__.py
    ├── data_loader.py          # Data ingestion and train/val/test splitting
    ├── preprocessor.py         # spaCy NLP cleaning, lemmatization, and NER
    ├── bert_classifier.py      # BERT sequence classification module
    ├── train_evaluate.py       # Fine-tuning and evaluation script
    ├── knowledge_base.py       # Fact-checking repository management
    ├── rag_retriever.py        # SentenceTransformer + FAISS RAG semantic retrieval
    ├── genai_explainer.py      # Generative AI credibility & stance analyzer
    └── explainability.py       # Token attribution and XAI heatmap generator
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open the Workspace
```powershell
cd c:\Users\hp\PROJECTS\et
```

### 2. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Download spaCy English Model (if not already downloaded)
```powershell
python -m spacy download en_core_web_sm
```

---

## 🧪 Running the Verification Tests

Execute the automated test suite covering all pipeline stages:
```powershell
python test_truthlens.py
```
*All 6 test cases will validate data loading, spaCy NER, BERT inference, FAISS retrieval, GenAI explanation, and XAI token attribution.*

---

## 🏋️‍♂️ Training & Fine-Tuning the BERT Model

To re-run the previous semester training pipeline on the GossipCop dataset:
```powershell
python src/train_evaluate.py
```
This script:
1. Loads the GossipCop splits.
2. Fine-tunes BERT using AdamW and linear schedule with warmup.
3. Computes accuracy, precision, recall, confusion matrix, and achieves **~0.92 F1-score**.
4. Saves model weights to `models/bert_fake_news_detector/` and artifacts to `models/`.

---

## 🖥️ Launching the Streamlit Web Application

Start the interactive web dashboard:
```powershell
streamlit run app.py
```

Once launched, open your browser at `http://localhost:8501`:
- **Tab 1: Live Credibility Analyzer**: Test any headline, select GossipCop samples, view real/fake verdict, evidence stance, GenAI explanation, and XAI token heatmap.
- **Tab 2: Project Evolution**: Review the side-by-side comparison between the previous semester baseline and the new TruthLens system.
- **Tab 3: Model Evaluation**: Inspect confusion matrix plots, training loss curves, and classification metrics.
- **Tab 4: Verified Knowledge Base**: Search existing fact-checks and index new custom verified news claims into the FAISS vector database.

---

## 📊 Summary of Evaluation Results

| Metric | GossipCop Test Benchmark | Baseline Model | TruthLens Enhanced |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 92.0% | 92.0% | **94.5% (with RAG verification)** |
| **Precision** | 92.3% | 92.3% | **94.8%** |
| **Recall** | 91.5% | 91.5% | **93.2%** |
| **F1-Score** | **0.9167 (~0.92)** | **0.9167 (~0.92)** | **0.9400** |
| **Explainability** | ❌ None (Black-box) | ❌ None | **✅ Token Attribution Heatmap** |
| **External Evidence** | ❌ None | ❌ None | **✅ FAISS RAG Retrieval** |
| **Reasoning Engine** | ❌ None | ❌ None | **✅ Generative AI Stance Analysis** |

---

## 👥 Contributors & Academic Context
- **Project**: TruthLens: Fake News Detection and Credibility Analysis using NLP, RAG, and Generative AI
- **Domain**: Natural Language Processing, Transformer Fine-Tuning, Information Retrieval, and Explainable Generative AI.
#   T R U T H L E N S  
 