"""
TruthLens - Explainable AI (XAI) Suite
Implements three state-of-the-art XAI methods for Transformer Text Classification:
1. LIME (Local Interpretable Model-agnostic Explanations)
2. SHAP (SHapley Additive exPlanations based on cooperative game theory)
3. Leave-One-Out Feature Perturbation (Fast Empirical Attribution)

Includes a unified comparison engine benchmarking attribution consistency, speed, and word importance rankings.
"""

import os
import sys
import re
import html
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bert_classifier import BertNewsClassifier, MODEL_SAVE_DIR

# Import LIME
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Import SHAP
try:
    import shap
    from transformers import pipeline
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _render_html_attribution(attributions: List[Dict[str, Any]], prediction: str, method_name: str = "") -> str:
    """
    Renders an HTML heatmap visualization for any attribution list.
    Red tones for FAKE-inducing tokens, Green tones for REAL-inducing tokens.
    """
    is_fake = (prediction == "FAKE")
    primary_color = "231, 76, 60" if is_fake else "46, 204, 113"  # Red vs Green
    alt_color = "46, 204, 113" if is_fake else "231, 76, 60"

    rendered_tokens = []
    for item in attributions:
        word = html.escape(str(item.get("word", "")))
        if item.get("is_punct", False):
            rendered_tokens.append(word)
            continue

        score = float(item.get("score", 0.0))
        norm = max(0.0, min(1.0, abs(float(item.get("norm_score", 0.0)))))
        alpha = max(0.10, norm * 0.85) if norm > 0.05 else 0.0

        target_color = primary_color if score >= 0 else alt_color

        if alpha > 0.08:
            score_val = score
            span = (
                f"<span style='background-color: rgba({target_color}, {alpha:.2f}); "
                f"padding: 2px 6px; border-radius: 4px; margin: 1px 2px; font-weight: 500; display: inline-block;' "
                f"title='{method_name} Weight: {score_val:.4f}'>{word}</span>"
            )
        else:
            span = f"<span style='margin: 1px 2px;'>{word}</span>"

        rendered_tokens.append(span)

    badge = f"<div style='font-size: 0.78rem; font-weight: 700; color: #64748b; margin-bottom: 6px; text-transform: uppercase;'>{method_name} Token Heatmap</div>" if method_name else ""
    return f"<div style='line-height: 2.3; font-size: 1.02rem; padding: 14px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;'>{badge}{' '.join(rendered_tokens)}</div>"


# ==========================================
# 1. PERTURBATION EXPLAINER (BASELINE)
# ==========================================
class TokenExplainer:
    """
    Computes token-level importance attribution using leave-one-out masking.
    Fast empirical baseline that measures delta in prediction confidence.
    """

    def __init__(self, classifier: BertNewsClassifier):
        self.classifier = classifier
        self.method_name = "Perturbation (Leave-One-Out)"

    def explain_instance(self, text: str, max_tokens: int = 35) -> Dict[str, Any]:
        start_time = time.time()
        if not text or not text.strip():
            return {"method": self.method_name, "execution_time_ms": 0, "attributions": [], "top_influential_words": [], "html_visualization": ""}

        base_result = self.classifier.predict(text)
        pred_label_id = base_result["label_id"]
        pred_label_name = base_result["prediction"]
        base_prob = base_result["confidence"]

        words = re.findall(r'\b\w+\b|[^\w\s]', text)[:max_tokens]
        attributions = []

        for idx, word in enumerate(words):
            if not word.isalnum():
                attributions.append({"word": word, "score": 0.0, "is_punct": True})
                continue

            perturbed_text = " ".join(words[:idx] + words[idx + 1:])
            perturbed_result = self.classifier.predict(perturbed_text)

            if perturbed_result["label_id"] == pred_label_id:
                diff = base_prob - perturbed_result["confidence"]
            else:
                diff = base_prob + (1.0 - perturbed_result["confidence"])

            attributions.append({"word": word, "score": float(diff), "is_punct": False})

        # Normalize
        scores = [a["score"] for a in attributions if not a["is_punct"]]
        max_abs = max(max(map(abs, scores), default=1e-5), 1e-5)
        for a in attributions:
            a["norm_score"] = round(a["score"] / max_abs, 4) if not a["is_punct"] else 0.0

        salient = [a for a in attributions if not a["is_punct"]]
        salient.sort(key=lambda x: x["score"], reverse=True)
        top_words = [
            {"word": w["word"], "importance": round(w["score"], 4), "norm_score": w["norm_score"], "impact": "High" if w["score"] > 0.05 else "Moderate"}
            for w in salient[:6] if w["score"] > 0.005
        ]

        exec_time = round((time.time() - start_time) * 1000, 1)
        html_markup = _render_html_attribution(attributions, pred_label_name, self.method_name)

        return {
            "method": self.method_name,
            "prediction": pred_label_name,
            "base_confidence": base_prob,
            "execution_time_ms": exec_time,
            "attributions": attributions,
            "top_influential_words": top_words,
            "html_visualization": html_markup
        }


# ==========================================
# 2. LIME EXPLAINER
# ==========================================
class LimeExplainer:
    """
    Local Interpretable Model-agnostic Explanations (LIME) for Text.
    Generates sparse linear surrogate models around perturbed variations of the sentence.
    """

    def __init__(self, classifier: BertNewsClassifier):
        self.classifier = classifier
        self.method_name = "LIME (Surrogate Linear Model)"
        if LIME_AVAILABLE:
            self.explainer = LimeTextExplainer(
                class_names=["FAKE", "REAL"],
                split_expression=r'\W+',
                bow=True,
                random_state=42
            )

    def _predict_proba_batch(self, texts: List[str]) -> np.ndarray:
        probs = []
        for t in texts:
            res = self.classifier.predict(t)
            probs.append([res["fake_probability"], res["real_probability"]])
        return np.array(probs)

    def explain_instance(self, text: str, num_features: int = 8, num_samples: int = 80) -> Dict[str, Any]:
        start_time = time.time()
        if not LIME_AVAILABLE or not text or not text.strip():
            return {"method": self.method_name, "execution_time_ms": 0, "attributions": [], "top_influential_words": [], "html_visualization": ""}

        base_res = self.classifier.predict(text)
        pred_label_id = base_res["label_id"]
        pred_label_name = base_res["prediction"]

        # Run LIME explanation
        exp = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=self._predict_proba_batch,
            num_features=num_features,
            num_samples=num_samples,
            labels=[pred_label_id]
        )

        raw_weights = exp.as_list(label=pred_label_id)
        weight_dict = {w.lower(): score for w, score in raw_weights}

        words = re.findall(r'\b\w+\b|[^\w\s]', text)[:35]
        attributions = []
        max_abs = max([abs(s) for _, s in raw_weights], default=1e-5) or 1e-5

        for word in words:
            if not word.isalnum():
                attributions.append({"word": word, "score": 0.0, "is_punct": True})
                continue

            score = float(weight_dict.get(word.lower(), 0.0))
            norm_score = round(score / max_abs, 4)
            attributions.append({
                "word": word,
                "score": score,
                "norm_score": norm_score,
                "is_punct": False
            })

        top_words = [
            {
                "word": w,
                "importance": round(float(score), 4),
                "norm_score": round(float(score) / max_abs, 4),
                "impact": "Supports Prediction" if score > 0 else "Opposes Prediction"
            }
            for w, score in raw_weights[:6]
        ]

        exec_time = round((time.time() - start_time) * 1000, 1)
        html_markup = _render_html_attribution(attributions, pred_label_name, self.method_name)

        return {
            "method": self.method_name,
            "prediction": pred_label_name,
            "base_confidence": base_res["confidence"],
            "execution_time_ms": exec_time,
            "attributions": attributions,
            "top_influential_words": top_words,
            "html_visualization": html_markup
        }


# ==========================================
# 3. SHAP EXPLAINER
# ==========================================
class ShapExplainer:
    """
    SHapley Additive exPlanations (SHAP) for Text.
    Uses cooperative game theory to compute mathematically fair Shapley values
    measuring the marginal contribution of each token.
    """

    def __init__(self, model_dir: str = MODEL_SAVE_DIR):
        self.method_name = "SHAP (Shapley Game Theory Values)"
        self.pipe = None
        self.explainer = None

        if SHAP_AVAILABLE and os.path.exists(model_dir):
            try:
                self.pipe = pipeline(
                    "text-classification",
                    model=model_dir,
                    tokenizer=model_dir,
                    return_all_scores=True,
                    device=-1  # CPU inference
                )
                self.explainer = shap.Explainer(self.pipe)
            except Exception as e:
                print(f"SHAP initialization notice: {e}")

    def explain_instance(self, text: str, max_tokens: int = 35) -> Dict[str, Any]:
        start_time = time.time()
        if not SHAP_AVAILABLE or self.explainer is None or not text or not text.strip():
            return {"method": self.method_name, "execution_time_ms": 0, "attributions": [], "top_influential_words": [], "html_visualization": ""}

        # Truncate to max_tokens for speed
        words = text.split()[:max_tokens]
        truncated_text = " ".join(words)

        # Run SHAP computation with optimized max_evals
        shap_values = self.explainer([truncated_text], max_evals=64)

        # Find target predicted class
        preds = self.pipe([truncated_text])[0]
        if isinstance(preds, list):
            sorted_preds = sorted(preds, key=lambda x: x["score"], reverse=True)
            pred_label_name = sorted_preds[0]["label"]
            base_confidence = sorted_preds[0]["score"]
        elif isinstance(preds, dict):
            pred_label_name = preds.get("label", "FAKE")
            base_confidence = preds.get("score", 0.5)
        else:
            pred_label_name = "FAKE"
            base_confidence = 0.5

        pred_label_id = 0 if pred_label_name == "FAKE" else 1

        tokens = shap_values.data[0]
        # Token SHAP values for predicted class
        raw_vals = shap_values.values[0, :, pred_label_id]

        attributions = []
        token_scores = []

        for tok, val in zip(tokens, raw_vals):
            clean_tok = tok.strip()
            if not clean_tok or clean_tok in ["[CLS]", "[SEP]", "<s>", "</s>"]:
                continue
            is_punct = not bool(re.search(r'\w', clean_tok))
            score = float(val)
            attributions.append({
                "word": clean_tok,
                "score": score,
                "is_punct": is_punct
            })
            if not is_punct:
                token_scores.append((clean_tok, score))

        # Normalize scores
        max_abs = max([abs(s) for _, s in token_scores], default=1e-5) or 1e-5
        for a in attributions:
            a["norm_score"] = round(a["score"] / max_abs, 4) if not a["is_punct"] else 0.0

        # Aggregate subword tokens into word-level map
        word_map = {}
        for w, sc in token_scores:
            base_w = w.lower()
            word_map[base_w] = word_map.get(base_w, 0.0) + sc

        sorted_words = sorted(word_map.items(), key=lambda x: abs(x[1]), reverse=True)
        top_words = [
            {
                "word": w,
                "importance": round(float(score), 4),
                "norm_score": round(float(score) / max_abs, 4),
                "impact": "Supports Prediction" if score >= 0 else "Opposes Prediction"
            }
            for w, score in sorted_words[:6]
        ]

        exec_time = round((time.time() - start_time) * 1000, 1)
        html_markup = _render_html_attribution(attributions, pred_label_name, self.method_name)

        return {
            "method": self.method_name,
            "prediction": pred_label_name,
            "base_confidence": base_confidence,
            "execution_time_ms": exec_time,
            "attributions": attributions,
            "top_influential_words": top_words,
            "html_visualization": html_markup
        }


# ==========================================
# 4. UNIFIED XAI COMPARISON ENGINE
# ==========================================
class XAIComparator:
    """
    Benchmarks and compares LIME, SHAP, and Perturbation on the same text instance.
    Evaluates:
    - Speed (Latency in milliseconds)
    - Agreement Ratio (Top token overlap)
    - Token attribution rankings
    - Theoretical trade-offs
    """

    def __init__(self, classifier: BertNewsClassifier):
        self.classifier = classifier
        self.pert_explainer = TokenExplainer(classifier)
        self.lime_explainer = LimeExplainer(classifier)
        self.shap_explainer = ShapExplainer(MODEL_SAVE_DIR)

    def compare(self, text: str) -> Dict[str, Any]:
        """Runs all available explainers and compiles a comprehensive comparison."""
        results = {}

        # 1. Run Perturbation
        pert_res = self.pert_explainer.explain_instance(text)
        results["Perturbation"] = pert_res

        # 2. Run LIME
        lime_res = self.lime_explainer.explain_instance(text)
        results["LIME"] = lime_res

        # 3. Run SHAP
        shap_res = self.shap_explainer.explain_instance(text)
        results["SHAP"] = shap_res

        # 4. Compute Agreement Metric (Jaccard similarity of top-4 words)
        words_pert = set(w["word"].lower() for w in pert_res.get("top_influential_words", [])[:4])
        words_lime = set(w["word"].lower() for w in lime_res.get("top_influential_words", [])[:4])
        words_shap = set(w["word"].lower() for w in shap_res.get("top_influential_words", [])[:4])

        lime_shap_intersection = words_lime.intersection(words_shap)
        lime_shap_union = words_lime.union(words_shap)
        lime_shap_agreement = (
            round((len(lime_shap_intersection) / len(lime_shap_union)) * 100, 1)
            if lime_shap_union else 100.0
        )

        all_three_intersection = words_pert.intersection(words_lime).intersection(words_shap)

        # 5. Build Unified Token Importance Table
        all_unique_words = list(dict.fromkeys(
            [w["word"].lower() for w in pert_res.get("top_influential_words", [])] +
            [w["word"].lower() for w in lime_res.get("top_influential_words", [])] +
            [w["word"].lower() for w in shap_res.get("top_influential_words", [])]
        ))[:6]

        pert_map = {w["word"].lower(): w["norm_score"] for w in pert_res.get("top_influential_words", [])}
        lime_map = {w["word"].lower(): w["norm_score"] for w in lime_res.get("top_influential_words", [])}
        shap_map = {w["word"].lower(): w["norm_score"] for w in shap_res.get("top_influential_words", [])}

        comparison_table = []
        for word in all_unique_words:
            comparison_table.append({
                "Token": word,
                "Perturbation Score": pert_map.get(word, 0.0),
                "LIME Score": lime_map.get(word, 0.0),
                "SHAP Score": shap_map.get(word, 0.0)
            })

        # 6. Methodological Breakdown
        methodology_comparison = [
            {
                "Dimension": "Mathematical Principle",
                "LIME": "Local Sparse Linear Regression surrogate around perturbed samples",
                "SHAP": "Cooperative Game Theory (Shapley Values from coalitions of tokens)",
                "Perturbation": "Direct empirical confidence shift via Leave-One-Out masking"
            },
            {
                "Dimension": "Axiomatic Guarantees",
                "LIME": "Heuristic / Local fidelity (sampling-dependent, stochastic)",
                "SHAP": "Provable axioms: Efficiency, Symmetry, Dummy player, Additivity",
                "Perturbation": "Direct empirical measurement, but ignores token interaction coalitions"
            },
            {
                "Dimension": "Inference Speed",
                "LIME": f"Moderate (~{lime_res.get('execution_time_ms', 0)} ms)",
                "SHAP": f"Heavier (~{shap_res.get('execution_time_ms', 0)} ms)",
                "Perturbation": f"Fastest (~{pert_res.get('execution_time_ms', 0)} ms)"
            },
            {
                "Dimension": "Key Advantage",
                "LIME": "Model-agnostic, intuitive word weights, fast surrogate fitting",
                "SHAP": "Mathematically rigorous, unique solution with theoretical fairness",
                "Perturbation": "Extremely lightweight, zero overhead, highly interpretable"
            }
        ]

        return {
            "individual_results": results,
            "comparison_table": comparison_table,
            "methodology_comparison": methodology_comparison,
            "lime_shap_agreement_percent": lime_shap_agreement,
            "consensus_words": list(lime_shap_intersection) or list(all_three_intersection) or (all_unique_words[:2] if all_unique_words else []),
            "latency": {
                "Perturbation": pert_res.get("execution_time_ms", 0),
                "LIME": lime_res.get("execution_time_ms", 0),
                "SHAP": shap_res.get("execution_time_ms", 0)
            }
        }


if __name__ == "__main__":
    classifier = BertNewsClassifier()
    comparator = XAIComparator(classifier)
    sample = "Jennifer Aniston pregnant with Brad Pitt secret miracle twin babies in Paris."
    print("Testing Full XAI Comparison Suite (LIME vs SHAP vs Perturbation)...")
    comp = comparator.compare(sample)
    print("\nLatency (ms):", comp["latency"])
    print("LIME & SHAP Agreement:", comp["lime_shap_agreement_percent"], "%")
    print("Consensus Tokens:", comp["consensus_words"])
    print("\nComparison Table:")
    for row in comp["comparison_table"]:
        print(f"  {row['Token']}: Pert={row['Perturbation Score']}, LIME={row['LIME Score']}, SHAP={row['SHAP Score']}")
