"""
TruthLens - Generative AI Credibility & Stance Analysis Engine
Synthesizes news content, BERT prediction, extracted entities, and RAG evidence
to generate transparent explanations and credibility assessments.
"""

import os
import re
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GenerativeCredibilityAnalyzer:
    """
    Dual-mode Generative AI Explainer:
    - Gemini API: When an API key is provided
    - Built-in Local Reasoning: Fully offline, rule-and-semantic NLI reasoning engine
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.is_gemini_active = False

        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                self.is_gemini_active = True
            except Exception as e:
                print(f"Gemini initialization warning: {e}. Defaulting to built-in local engine.")

    def set_api_key(self, api_key: str):
        """Allows dynamic setting of Gemini API key from UI."""
        if not api_key:
            self.is_gemini_active = False
            return
        if GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                self.api_key = api_key
                self.is_gemini_active = True
            except Exception as e:
                print(f"Failed to update Gemini API key: {e}")
                self.is_gemini_active = False

    def analyze_credibility(
        self,
        news_text: str,
        bert_result: Dict[str, Any],
        retrieved_evidence: List[Dict[str, Any]],
        entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main credibility assessment method combining all signals.
        """
        # 1. Determine Evidence Stance
        stance, stance_details = self._determine_stance(news_text, retrieved_evidence)

        # 2. Compute Integrated Credibility Score (0 - 100%)
        credibility_score, credibility_rating = self._calculate_credibility_score(
            bert_result=bert_result,
            stance=stance,
            top_evidence=retrieved_evidence[0] if retrieved_evidence else None
        )

        # 3. Generate Evidence-Based Explanation
        if self.is_gemini_active:
            try:
                explanation = self._generate_gemini_explanation(
                    news_text=news_text,
                    bert_result=bert_result,
                    retrieved_evidence=retrieved_evidence,
                    stance=stance,
                    credibility_score=credibility_score
                )
                source_engine = "Gemini 1.5 Flash (Generative AI)"
            except Exception as e:
                explanation = self._generate_local_explanation(
                    news_text=news_text,
                    bert_result=bert_result,
                    retrieved_evidence=retrieved_evidence,
                    stance=stance,
                    credibility_score=credibility_score
                )
                source_engine = f"Built-in Local Neural Reasoning (Gemini fallback: {str(e)[:40]}...)"
        else:
            explanation = self._generate_local_explanation(
                news_text=news_text,
                bert_result=bert_result,
                retrieved_evidence=retrieved_evidence,
                stance=stance,
                credibility_score=credibility_score
            )
            source_engine = "Built-in Local Fact-Checking Reasoning Engine"

        return {
            "stance": stance,
            "stance_label": self._format_stance_label(stance),
            "credibility_score": credibility_score,
            "credibility_rating": credibility_rating,
            "ai_explanation": explanation,
            "reasoning_engine": source_engine,
            "evidence_count": len(retrieved_evidence),
            "stance_details": stance_details
        }

    def _determine_stance(
        self,
        news_text: str,
        retrieved_evidence: List[Dict[str, Any]]
    ) -> tuple:
        """
        Determines whether the retrieved verified evidence:
        - SUPPORTS the claim
        - CONTRADICTS the claim
        - INSUFFICIENT_EVIDENCE
        """
        if not retrieved_evidence:
            return "INSUFFICIENT_EVIDENCE", "No relevant external fact-checks matched the claim in the knowledge base."

        top_doc = retrieved_evidence[0]
        sim_score = top_doc.get("similarity_score", 0.0)

        # If similarity is too low, evidence is insufficient
        if sim_score < 0.40:
            return "INSUFFICIENT_EVIDENCE", f"Top matched document has weak semantic similarity ({sim_score:.2f} < 0.40)."

        verdict_str = top_doc.get("verdict", "").upper()

        if "FALSE" in verdict_str or "DEBUNKED" in verdict_str or "HOAX" in verdict_str:
            return "CONTRADICTS", f"Fact-checking evidence from {top_doc.get('source')} officially debunks or refutes this claim."

        elif "TRUE" in verdict_str or "VERIFIED" in verdict_str or "CONFIRMED" in verdict_str:
            return "SUPPORTS", f"Fact-checking evidence from {top_doc.get('source')} confirms and corroborates this claim."

        return "INSUFFICIENT_EVIDENCE", "Evidence retrieved is neutral or requires deeper investigation."

    def _format_stance_label(self, stance: str) -> str:
        mapping = {
            "CONTRADICTS": "Contradicts the Claim (Debunked by Evidence)",
            "SUPPORTS": "Supports the Claim (Verified by Official Sources)",
            "INSUFFICIENT_EVIDENCE": "Does Not Provide Enough Evidence to Verify"
        }
        return mapping.get(stance, stance)

    def _calculate_credibility_score(
        self,
        bert_result: Dict[str, Any],
        stance: str,
        top_evidence: Optional[Dict[str, Any]]
    ) -> tuple:
        """
        Synthesizes a 0-100 credibility index combining BERT probability and RAG evidence stance.
        """
        # Base score from BERT real probability (0 to 100)
        bert_real_prob = bert_result.get("real_probability", 0.5)
        base_score = bert_real_prob * 100.0

        if stance == "CONTRADICTS":
            # Strongly penalize score when verified fact check refutes the claim
            sim = top_evidence.get("similarity_score", 0.7) if top_evidence else 0.7
            credibility_score = max(5.0, base_score * (1.0 - sim) * 0.5)
        elif stance == "SUPPORTS":
            # Boost score when verified fact check supports the claim
            sim = top_evidence.get("similarity_score", 0.7) if top_evidence else 0.7
            credibility_score = min(98.0, base_score + (100.0 - base_score) * sim * 0.8)
        else:
            # Neutral / insufficient: rely primarily on BERT model confidence
            credibility_score = base_score

        credibility_score = round(credibility_score, 1)

        # Rating Category
        if credibility_score >= 75.0:
            rating = "High Credibility"
        elif credibility_score >= 50.0:
            rating = "Moderate Credibility"
        elif credibility_score >= 25.0:
            rating = "Questionable / Likely Misinformation"
        else:
            rating = "Debunked / Severe Misinformation"

        return credibility_score, rating

    def _generate_local_explanation(
        self,
        news_text: str,
        bert_result: Dict[str, Any],
        retrieved_evidence: List[Dict[str, Any]],
        stance: str,
        credibility_score: float
    ) -> str:
        """
        Generates an evidence-backed narrative explanation offline.
        """
        prediction = bert_result.get("prediction", "UNKNOWN")
        confidence_pct = round(bert_result.get("confidence", 0.5) * 100, 1)

        if not retrieved_evidence or stance == "INSUFFICIENT_EVIDENCE":
            return (
                f"The BERT classification model analyzed the linguistic and contextual patterns of this content "
                f"and flagged it as {prediction} with {confidence_pct}% model confidence. "
                f"However, our verified knowledge base currently lacks a direct external fact-check regarding this specific claim. "
                f"Users are advised to cross-reference with primary news agencies."
            )

        top_doc = retrieved_evidence[0]
        evidence_source = top_doc.get("source", "Verified Fact-Checking Records")
        evidence_summary = top_doc.get("fact_check_summary", "")
        relevance_pct = top_doc.get("relevance_percent", 0.0)

        if stance == "CONTRADICTS":
            return (
                f"The content is classified as {prediction} with {confidence_pct}% model confidence, and our RAG "
                f"retrieval engine identified verified evidence from {evidence_source} (Relevance: {relevance_pct}%) that CONTRADICTS this claim. "
                f"According to the investigative findings: \"{evidence_summary}\" "
                f"Consequently, this claim has been assessed as fabricated or misleading, resulting in a low credibility score of {credibility_score}%."
            )
        elif stance == "SUPPORTS":
            return (
                f"The content is classified as {prediction} with {confidence_pct}% model confidence, and our RAG "
                f"retrieval engine identified official documentation from {evidence_source} (Relevance: {relevance_pct}%) that SUPPORTS this claim. "
                f"Official reporting confirms: \"{evidence_summary}\" "
                f"The consistent alignment between transformer classification and primary fact-checks yields a high credibility score of {credibility_score}%."
            )
        else:
            return (
                f"The content was evaluated as {prediction} ({confidence_pct}% confidence). "
                f"Retrieved information from {evidence_source} does not provide conclusive verification. "
                f"Overall credibility is rated at {credibility_score}%."
            )

    def _generate_gemini_explanation(
        self,
        news_text: str,
        bert_result: Dict[str, Any],
        retrieved_evidence: List[Dict[str, Any]],
        stance: str,
        credibility_score: float
    ) -> str:
        """
        Uses Google Gemini to synthesize a human-like, deep explanation based on provided evidence.
        """
        evidence_snippets = "\n".join([
            f"- [{doc.get('source')} | Verdict: {doc.get('verdict')}]: {doc.get('fact_check_summary')}"
            for doc in retrieved_evidence[:2]
        ])

        prompt = f"""
You are TruthLens AI, an objective fact-checking assistant analyzing news credibility.
A user has submitted the following news item:
"{news_text}"

Analysis Context:
- BERT Classifier Prediction: {bert_result.get('prediction')} (Confidence: {bert_result.get('confidence')*100:.1f}%)
- Evidence Stance: {stance}
- Calculated Credibility Score: {credibility_score}%

Retrieved Verified Evidence from Knowledge Base:
{evidence_snippets if evidence_snippets else "No direct fact-check found in database."}

Instructions:
Provide a concise, 3 to 4 sentence objective explanation explaining:
1. Whether the claim is authentic or fabricated.
2. How the retrieved evidence from the verified sources supports or contradicts the claim.
3. Why this yields the overall credibility assessment.
Be factual, professional, and clear. Avoid speculation.
"""
        response = self.gemini_model.generate_content(prompt)
        return response.text.strip()


if __name__ == "__main__":
    analyzer = GenerativeCredibilityAnalyzer()
    sample_news = "Jennifer Aniston pregnant with Brad Pitt's miracle twin babies after secret Paris wedding."
    sample_bert = {"prediction": "FAKE", "confidence": 0.94, "real_probability": 0.06, "fake_probability": 0.94}
    sample_evidence = [{
        "title": "Fact Check: Jennifer Aniston and Brad Pitt Did Not Secretly Remarry",
        "verdict": "FALSE / DEBUNKED",
        "source": "Snopes Fact Check",
        "similarity_score": 0.72,
        "relevance_percent": 72.0,
        "fact_check_summary": "Representatives for Jennifer Aniston and Brad Pitt have clarified the former spouses never remarried in France and are not expecting children."
    }]
    res = analyzer.analyze_credibility(sample_news, sample_bert, sample_evidence, {})
    print("GenAI Credibility Analysis Test:")
    print("Stance:", res["stance_label"])
    print("Credibility Score:", res["credibility_score"])
    print("Rating:", res["credibility_rating"])
    print("Explanation:", res["ai_explanation"])
