"""
TruthLens - Automated Verification Test Suite
Tests every individual pipeline component:
1. Data loading & splitting
2. spaCy preprocessing, lemmatization, and NER
3. Fine-tuned BERT classification & confidence scoring
4. SentenceTransformer + FAISS RAG semantic retrieval
5. Generative AI credibility & stance synthesis
6. Explainable AI (XAI) token attribution
"""

import os
import sys
import unittest

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_gossipcop_splits
from src.preprocessor import clean_text, lemmatize_text, extract_entities, full_nlp_pipeline
from src.bert_classifier import BertNewsClassifier
from src.rag_retriever import RAGRetriever
from src.genai_explainer import GenerativeCredibilityAnalyzer
from src.explainability import TokenExplainer, LimeExplainer, ShapExplainer, XAIComparator


class TestTruthLensPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 60)
        print("INITIALIZING TRUTHLENS TEST SUITE")
        print("=" * 60)
        cls.classifier = BertNewsClassifier()
        cls.retriever = RAGRetriever()
        cls.genai = GenerativeCredibilityAnalyzer()
        cls.explainer = TokenExplainer(cls.classifier)

    def test_01_data_loader(self):
        print("\n[TEST 1] Verifying GossipCop Dataset and Splits...")
        train_df, val_df, test_df, full_df = load_gossipcop_splits()
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(val_df), 0)
        self.assertGreater(len(test_df), 0)
        self.assertIn("full_text", train_df.columns)
        self.assertIn("label", train_df.columns)
        print(f"  Passed! Total: {len(full_df)}, Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    def test_02_spacy_preprocessing_and_ner(self):
        print("\n[TEST 2] Verifying spaCy NLP Preprocessing and NER...")
        sample_text = "Prince Harry and Meghan Markle officially visited London on May 10th."
        pipeline_res = full_nlp_pipeline(sample_text)
        
        self.assertIn("cleaned_text", pipeline_res)
        self.assertIn("lemmatized_text", pipeline_res)
        self.assertIn("ner", pipeline_res)
        
        ner = pipeline_res["ner"]
        print(f"  Extracted Entities: {ner['entities_by_type']}")
        self.assertTrue(any("Harry" in p or "Meghan" in p for p in ner["people"]) or len(ner["all_entities"]) > 0)
        self.assertGreater(len(pipeline_res["lemmatized_text"]), 0)
        print("  Passed! spaCy NER and lemmatization working accurately.")

    def test_03_bert_classifier(self):
        print("\n[TEST 3] Verifying BERT Sequence Classification & Confidence...")
        sample_news = "Brad Pitt and Jennifer Aniston secretly remarried in Paris with twin babies."
        pred = self.classifier.predict(sample_news)
        
        self.assertIn("prediction", pred)
        self.assertIn("confidence", pred)
        self.assertIn("real_probability", pred)
        self.assertIn("fake_probability", pred)
        self.assertIn(pred["prediction"], ["REAL", "FAKE"])
        self.assertGreaterEqual(pred["confidence"], 0.5)
        self.assertAlmostEqual(pred["real_probability"] + pred["fake_probability"], 1.0, places=3)
        print(f"  Prediction: {pred['prediction']} (Confidence: {pred['confidence']:.3f})")
        print("  Passed! BERT inference calibrated.")

    def test_04_rag_faiss_retrieval(self):
        print("\n[TEST 4] Verifying SentenceTransformer + FAISS Vector Search...")
        query = "Did Lady Gaga move in with Bradley Cooper after his split from Irina?"
        results = self.retriever.retrieve(query, top_k=2)
        
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertIn("title", top_match)
        self.assertIn("verdict", top_match)
        self.assertIn("similarity_score", top_match)
        self.assertGreater(top_match["similarity_score"], 0.4)
        print(f"  Top Matched Fact-Check: '{top_match['title']}' (Score: {top_match['similarity_score']})")
        print("  Passed! FAISS retrieval returned topically aligned verified fact-check.")

    def test_05_genai_credibility_and_stance(self):
        print("\n[TEST 5] Verifying Generative AI Credibility & Stance Synthesis...")
        sample_claim = "Lady Gaga secretly moved into Bradley Cooper's New York home."
        bert_result = {"prediction": "FAKE", "confidence": 0.88, "real_probability": 0.12, "fake_probability": 0.88}
        evidence = [{
            "title": "Fact Check: Lady Gaga Did Not Move In With Bradley Cooper",
            "verdict": "FALSE / DEBUNKED",
            "source": "PolitiFact Entertainment",
            "similarity_score": 0.74,
            "relevance_percent": 74.0,
            "fact_check_summary": "Lady Gaga directly refuted dating rumors on Jimmy Kimmel Live and Elle Magazine."
        }]
        
        analysis = self.genai.analyze_credibility(
            news_text=sample_claim,
            bert_result=bert_result,
            retrieved_evidence=evidence,
            entities={"people": ["Lady Gaga", "Bradley Cooper"]}
        )
        
        self.assertIn("stance", analysis)
        self.assertEqual(analysis["stance"], "CONTRADICTS")
        self.assertIn("credibility_score", analysis)
        self.assertLess(analysis["credibility_score"], 40.0)
        self.assertIn("ai_explanation", analysis)
        self.assertGreater(len(analysis["ai_explanation"]), 20)
        print(f"  Stance: {analysis['stance_label']}")
        print(f"  Credibility Score: {analysis['credibility_score']}% ({analysis['credibility_rating']})")
        print(f"  Explanation: {analysis['ai_explanation'][:100]}...")
        print("  Passed! Generative AI synthesized evidence-based explanation.")

    def test_06_explainability_token_attribution(self):
        print("\n[TEST 6] Verifying Explainable AI (XAI) Token Attribution...")
        sample_claim = "Shocking rumor claims Brad Pitt secret marriage in Paris."
        xai_res = self.explainer.explain_instance(sample_claim)
        
        self.assertIn("attributions", xai_res)
        self.assertIn("html_visualization", xai_res)
        self.assertGreater(len(xai_res["attributions"]), 0)
        print(f"  Influential Tokens Identified: {[t['word'] for t in xai_res['top_influential_words']]}")
        print("  Passed! Token perturbation explainability successfully rendered.")

    def test_07_lime_explainer(self):
        print("\n[TEST 7] Verifying LIME Text Explainer...")
        lime_explainer = LimeExplainer(self.classifier)
        sample = "Shocking rumor claims Brad Pitt secret marriage in Paris."
        lime_res = lime_explainer.explain_instance(sample, num_features=5, num_samples=30)
        self.assertIn("attributions", lime_res)
        self.assertIn("top_influential_words", lime_res)
        self.assertGreater(len(lime_res["top_influential_words"]), 0)
        print(f"  LIME Top Words: {[w['word'] for w in lime_res['top_influential_words']]}")
        print(f"  LIME Latency: {lime_res['execution_time_ms']} ms")
        print("  Passed! LIME surrogate linear model attribution successfully computed.")

    def test_08_shap_explainer(self):
        print("\n[TEST 8] Verifying SHAP Explainer (Game Theory Shapley Values)...")
        shap_explainer = ShapExplainer()
        sample = "Shocking rumor claims Brad Pitt secret marriage in Paris."
        shap_res = shap_explainer.explain_instance(sample, max_tokens=15)
        self.assertIn("attributions", shap_res)
        self.assertIn("top_influential_words", shap_res)
        self.assertGreater(len(shap_res["attributions"]), 0)
        print(f"  SHAP Top Words: {[w['word'] for w in shap_res['top_influential_words']]}")
        print(f"  SHAP Latency: {shap_res['execution_time_ms']} ms")
        print("  Passed! SHAP game-theoretic token attribution successfully computed.")

    def test_09_xai_comparison(self):
        print("\n[TEST 9] Verifying Unified XAI Comparison Suite (LIME vs SHAP vs Perturbation)...")
        comparator = XAIComparator(self.classifier)
        sample = "Jennifer Aniston pregnant with Brad Pitt secret miracle twins in Paris."
        comp = comparator.compare(sample)
        self.assertIn("individual_results", comp)
        self.assertIn("comparison_table", comp)
        self.assertIn("methodology_comparison", comp)
        self.assertIn("latency", comp)
        self.assertIn("lime_shap_agreement_percent", comp)
        print(f"  Comparison Latencies (ms): {comp['latency']}")
        print(f"  LIME & SHAP Agreement: {comp['lime_shap_agreement_percent']}%")
        print(f"  Consensus Tokens: {comp['consensus_words']}")
        print("  Passed! Unified XAI benchmark comparison successfully executed.")


if __name__ == "__main__":
    unittest.main()
