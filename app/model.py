"""
Model loading and prediction logic
"""
import logging
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalAssertionModel:
    """
    Wrapper for the clinical assertion classification model.
    Loads model once during initialization and caches for predictions.
    """

    MODEL_NAME = "bvanaken/clinical-assertion-negation-bert"

    def __init__(self):
        """Initialize and load the model and tokenizer"""
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        """Load the pre-trained model and tokenizer"""
        try:
            logger.info(f"Loading model {self.MODEL_NAME}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, sentence: str) -> Dict[str, any]:
        """
        Predict assertion status for a single sentence

        Args:
            sentence: Clinical text to classify

        Returns:
            Dictionary with 'label' and 'score'
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")

        # Tokenize input
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # Get prediction
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()

        # Map class index to label
        label = self.model.config.id2label[predicted_class]

        return {"label": label, "score": round(confidence, 4)}

    def predict_batch(self, sentences: List[str]) -> List[Dict[str, any]]:
        """
        Predict assertion status for multiple sentences

        Args:
            sentences: List of clinical texts to classify

        Returns:
            List of dictionaries with 'label' and 'score'
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")

        # Tokenize all inputs
        inputs = self.tokenizer(sentences, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # Get predictions for all sentences
            predicted_classes = torch.argmax(probs, dim=-1)
            confidences = torch.max(probs, dim=-1).values

        # Format results
        results = []
        for pred_class, confidence in zip(predicted_classes, confidences):
            label = self.model.config.id2label[pred_class.item()]
            results.append({"label": label, "score": round(confidence.item(), 4)})

        return results

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None


# Global model instance (loaded once at startup)
model_instance = None


def get_model() -> ClinicalAssertionModel:
    """Get or initialize the global model instance"""
    global model_instance
    if model_instance is None:
        model_instance = ClinicalAssertionModel()
    return model_instance
