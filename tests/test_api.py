"""
Unit tests for the Clinical Assertion Classification API
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint"""

    def test_health_check(self):
        """Test that health endpoint returns success"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "model_loaded" in data
        assert "model_name" in data


class TestRootEndpoint:
    """Tests for root endpoint"""

    def test_root(self):
        """Test that root endpoint returns API information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "health" in data


class TestPredictionEndpoint:
    """Tests for single prediction endpoint"""

    def test_predict_absent_denial(self):
        """Test: 'The patient denies chest pain.' -> ABSENT"""
        response = client.post("/predict", json={"sentence": "The patient denies chest pain."})
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "score" in data
        assert data["label"] == "ABSENT"
        assert 0.0 <= data["score"] <= 1.0

    def test_predict_present_history(self):
        """Test: 'He has a history of hypertension.' -> PRESENT"""
        response = client.post("/predict", json={"sentence": "He has a history of hypertension."})
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "PRESENT"
        assert 0.0 <= data["score"] <= 1.0

    def test_predict_conditional(self):
        """Test prediction for conditional sentence"""
        response = client.post("/predict", json={"sentence": "If the patient experiences dizziness, reduce the dosage."})
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "score" in data
        assert data["label"] in ["PRESENT", "ABSENT", "CONDITIONAL", "POSSIBLE", "HYPOTHETICAL"]
        assert 0.0 <= data["score"] <= 1.0

    def test_predict_absent_no_signs(self):
        """Test: 'No signs of pneumonia were observed.' -> ABSENT"""
        response = client.post("/predict", json={"sentence": "No signs of pneumonia were observed."})
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "ABSENT"
        assert 0.0 <= data["score"] <= 1.0

    def test_predict_empty_sentence(self):
        """Test that empty sentence returns validation error"""
        response = client.post("/predict", json={"sentence": ""})
        assert response.status_code == 422  # Validation error

    def test_predict_missing_sentence(self):
        """Test that missing sentence field returns validation error"""
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error

    def test_predict_performance(self):
        """Test that prediction completes in under 500ms"""
        import time

        start = time.time()
        response = client.post("/predict", json={"sentence": "The patient denies chest pain."})
        elapsed = (time.time() - start) * 1000

        assert response.status_code == 200
        # Note: This may fail on slow systems, adjust as needed
        assert elapsed < 500, f"Prediction took {elapsed}ms, expected < 500ms"


class TestBatchPredictionEndpoint:
    """Tests for batch prediction endpoint"""

    def test_batch_predict_multiple_sentences(self):
        """Test batch prediction with multiple sentences"""
        sentences = [
            "The patient denies chest pain.",
            "He has a history of hypertension.",
            "If the patient experiences dizziness, reduce the dosage.",
            "No signs of pneumonia were observed.",
        ]

        response = client.post("/predict/batch", json={"sentences": sentences})

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == len(sentences)

        # Verify each prediction has required fields
        for prediction in data["predictions"]:
            assert "label" in prediction
            assert "score" in prediction
            assert 0.0 <= prediction["score"] <= 1.0

    def test_batch_predict_single_sentence(self):
        """Test batch prediction with a single sentence"""
        response = client.post("/predict/batch", json={"sentences": ["The patient denies chest pain."]})

        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["label"] == "ABSENT"

    def test_batch_predict_empty_list(self):
        """Test that empty sentence list returns validation error"""
        response = client.post("/predict/batch", json={"sentences": []})
        assert response.status_code == 422  # Validation error

    def test_batch_predict_expected_labels(self):
        """Test batch prediction returns valid labels"""
        test_cases = [
            "The patient denies chest pain.",
            "He has a history of hypertension.",
            "If the patient experiences dizziness, reduce the dosage.",
            "No signs of pneumonia were observed.",
        ]

        response = client.post("/predict/batch", json={"sentences": test_cases})

        assert response.status_code == 200
        data = response.json()
        predictions = data["predictions"]

        valid_labels = ["PRESENT", "ABSENT", "CONDITIONAL", "POSSIBLE", "HYPOTHETICAL"]
        for i, prediction in enumerate(predictions):
            assert "label" in prediction, f"Sentence {i}: Missing label"
            assert "score" in prediction, f"Sentence {i}: Missing score"
            assert prediction["label"] in valid_labels, f"Sentence {i}: Invalid label {prediction['label']}"
            assert 0.0 <= prediction["score"] <= 1.0, f"Sentence {i}: Invalid score {prediction['score']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
