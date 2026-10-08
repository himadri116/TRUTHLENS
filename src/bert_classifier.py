"""
TruthLens - BERT Classifier Module (Previous Semester Core)
Implements fine-tuned Transformer-based sequence classification for Fake vs. Real News.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, Tuple, Optional

# Standard model identifier
DEFAULT_BASE_MODEL = "distilbert-base-uncased"
MODEL_SAVE_DIR = "models/bert_fake_news_detector"


class BertNewsClassifier:
    """
    BERT-based binary classification model for fake news detection.
    Predicts:
      0 -> FAKE News
      1 -> REAL News
    """

    def __init__(self, model_name_or_path: str = DEFAULT_BASE_MODEL, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_path = model_name_or_path
        self.label_map = {0: "FAKE", 1: "REAL"}
        
        # Load tokenizer and model
        self._load_model()

    def _load_model(self):
        """Loads pretrained or fine-tuned model and tokenizer."""
        # Check if fine-tuned weights exist in local models directory
        if os.path.exists(MODEL_SAVE_DIR) and os.path.exists(os.path.join(MODEL_SAVE_DIR, "config.json")):
            target_path = MODEL_SAVE_DIR
        else:
            target_path = self.model_path

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(target_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                target_path,
                num_labels=2,
                id2label=self.label_map,
                label2id={"FAKE": 0, "REAL": 1}
            )
        except Exception:
            # Fallback to default base
            self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                DEFAULT_BASE_MODEL,
                num_labels=2,
                id2label=self.label_map,
                label2id={"FAKE": 0, "REAL": 1}
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, max_length: int = 256) -> Dict[str, Any]:
        """
        Runs inference on input text.
        Returns:
          - prediction: 'REAL' or 'FAKE'
          - label_id: 1 or 0
          - confidence: float (0.0 to 1.0)
          - real_probability: float
          - fake_probability: float
          - token_count: int
        """
        if not text or not text.strip():
            return {
                "prediction": "UNKNOWN",
                "label_id": -1,
                "confidence": 0.0,
                "real_probability": 0.5,
                "fake_probability": 0.5,
                "token_count": 0
            }

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        fake_prob = float(probs[0])
        real_prob = float(probs[1])

        label_id = int(torch.argmax(logits, dim=-1).item())
        prediction = self.label_map[label_id]
        confidence = float(max(fake_prob, real_prob))

        return {
            "prediction": prediction,
            "label_id": label_id,
            "confidence": round(confidence, 4),
            "real_probability": round(real_prob, 4),
            "fake_probability": round(fake_prob, 4),
            "token_count": int(inputs["input_ids"].shape[1])
        }

    def save(self, output_dir: str = MODEL_SAVE_DIR):
        """Saves tokenizer and model weights."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    classifier = BertNewsClassifier()
    sample = "Jennifer Aniston and Justin Theroux announce their mutual divorce after two years."
    result = classifier.predict(sample)
    print("Inference Test:")
    print(result)
