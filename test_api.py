"""
Automated Verification Test Suite for BhoomiSetu ML API
Tests Authentication, Endpoints, Inference, Batching, and SHAP Explanations
"""

from fastapi.testclient import TestClient
from app import app
import key_manager

client = TestClient(app)

def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    assert "accuracy" in data
    print("[PASS] Health endpoint returned 200 OK and model is online.")

def test_unauthorized_access():
    # Try calling predict without key
    response = client.post("/api/v1/predict", json={"state": "Maharashtra"})
    assert response.status_code == 401
    assert "Missing API Key" in response.json()["detail"]
    print("[PASS] Unauthorized request rejected with 401.")

def test_invalid_key():
    response = client.post(
        "/api/v1/predict",
        headers={"X-API-Key": "bs_live_invalid_key_1234567890"},
        json={"state": "Maharashtra"}
    )
    assert response.status_code == 401
    assert "Invalid or revoked API Key" in response.json()["detail"]
    print("[PASS] Invalid key rejected with 401.")

def test_prediction_with_valid_key():
    valid_key = key_manager.get_or_create_default_key()
    payload = {
        "project_id": "TEST_PRJ_01",
        "state": "Maharashtra",
        "district": "Pune",
        "project_type": "Highways",
        "acquisition_stage": "Section 19 (Declaration)",
        "project_cost": 1450.0,
        "land_acquired_pct": 35.0,
        "possession_pct": 25.0,
        "compensation_pending_pct": 65.0,
        "court_case_count": 8,
        "legal_case_count": 12,
        "days_in_current_stage": 120
    }
    response = client.post(
        "/api/v1/predict",
        headers={"X-API-Key": valid_key},
        json=payload
    )
    assert response.status_code == 200
    res = response.json()
    assert "delay_probability" in res
    assert "risk_tier" in res
    assert res["risk_tier"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert "recommended_action" in res
    print(f"[PASS] Prediction successful: {res['risk_tier']} Risk ({res['delay_probability']*100:.1f}%)")

def test_bearer_token_auth():
    valid_key = key_manager.get_or_create_default_key()
    payload = {"project_id": "TEST_BEARER", "state": "Gujarat"}
    response = client.post(
        "/api/v1/predict",
        headers={"Authorization": f"Bearer {valid_key}"},
        json=payload
    )
    assert response.status_code == 200
    assert "delay_probability" in response.json()
    print("[PASS] Bearer token authorization works successfully.")

def test_batch_prediction():
    valid_key = key_manager.get_or_create_default_key()
    payload = {
        "projects": [
            {"project_id": "BATCH_01", "state": "Haryana", "court_case_count": 0, "land_acquired_pct": 95.0},
            {"project_id": "BATCH_02", "state": "Bihar", "court_case_count": 15, "land_acquired_pct": 20.0}
        ]
    }
    response = client.post(
        "/api/v1/predict/batch",
        headers={"X-API-Key": valid_key},
        json=payload
    )
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    assert results[0]["project_id"] == "BATCH_01"
    assert results[1]["project_id"] == "BATCH_02"
    print(f"[PASS] Batch prediction successful for {len(results)} items.")

def test_explainability():
    valid_key = key_manager.get_or_create_default_key()
    payload = {
        "project_id": "TEST_EXPLAIN",
        "state": "Maharashtra",
        "court_case_count": 10,
        "compensation_pending_pct": 75.0
    }
    response = client.post(
        "/api/v1/explain",
        headers={"X-API-Key": valid_key},
        json=payload
    )
    assert response.status_code == 200
    res = response.json()
    assert "top_risk_drivers" in res
    assert len(res["top_risk_drivers"]) > 0
    top_driver = res["top_risk_drivers"][0]
    assert "feature" in top_driver
    assert "impact_score" in top_driver
    assert "direction" in top_driver
    print(f"[PASS] SHAP explainability successful: Top driver = {top_driver['feature']} ({top_driver['impact_score']})")

if __name__ == "__main__":
    print("\n--- RUNNING BHOOMISETU API TESTS ---")
    test_health()
    test_unauthorized_access()
    test_invalid_key()
    test_prediction_with_valid_key()
    test_bearer_token_auth()
    test_batch_prediction()
    test_explainability()
    print("\n[+] ALL 6 TESTS PASSED WITH 100% SUCCESS!\n")
