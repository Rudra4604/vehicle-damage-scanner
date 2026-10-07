"""Unit and integration tests for the FastAPI Vehicle Damage Detection backend.
"""
import hashlib
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
import pytest

from app.main import app

EXPECTED_BEST_PT_SHA256 = (
    "59f7a958e23cd84777c0785626e2a4ef6d24911051702d8cc488ef89ec2ac75f"
)


def get_file_sha256(path: Path) -> str:
    """Computes SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Session-scoped FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    """Path to known sample vehicle damage test image."""
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "000012.jpg"
    if fixture_path.exists():
        return fixture_path
    img_path = PROJECT_ROOT / "dataset" / "test" / "images" / "000012.jpg"
    assert img_path.exists(), f"Sample test image missing: {img_path}"
    return img_path


def test_root_endpoint_serves_frontend(client: TestClient):
    """Test GET / returns HTTP 200 and serves the frontend HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "<title>Vehicle Damage Scanner" in response.text
    assert 'id="dropzone"' in response.text
    assert 'id="view-upload"' in response.text
    assert 'id="view-results"' in response.text
    assert 'id="btn-analyze"' in response.text


def test_health_endpoint(client: TestClient):
    """Test GET /api/health returns operational status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "vehicle-damage-detection"
    assert "version" in data
    assert "device" in data
    assert isinstance(data.get("classes"), list)
    assert len(data["classes"]) == 6


def test_successful_image_upload_and_detection_structure(
    client: TestClient, sample_image_path: Path
):
    """Test POST /api/detect with valid image returns structured detection output."""
    with open(sample_image_path, "rb") as img_file:
        response = client.post(
            "/api/detect",
            files={"file": (sample_image_path.name, img_file, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()

    # 1. Verify required top-level JSON fields
    expected_fields = [
        "damage_status",
        "has_damage",
        "total_detections",
        "unique_categories",
        "average_confidence",
        "damage_area_percentage",
        "detections",
        "image_width",
        "image_height",
        "inference_time_ms",
        "annotated_image",
    ]
    for field in expected_fields:
        assert field in data, f"Missing required field '{field}' in response"

    # 2. Verify values and types
    assert data["damage_status"] == "Damage Detected"
    assert data["has_damage"] is True
    assert data["total_detections"] > 0
    assert data["image_width"] > 0
    assert data["image_height"] > 0
    assert isinstance(data["unique_categories"], list)
    assert 0.0 < data["average_confidence"] <= 1.0
    assert data["damage_area_percentage"] >= 0.0
    assert isinstance(data["detections"], list)
    assert len(data["detections"]) == data["total_detections"]

    # 3. Verify detection box structure
    det = data["detections"][0]
    for box_field in [
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "bounding_box_area",
    ]:
        assert box_field in det

    # 4. Verify annotated image path format
    assert data["annotated_image"].startswith("/api/results/")


def test_result_image_retrieval(client: TestClient, sample_image_path: Path):
    """Test retrieving an annotated image via GET /api/results/{filename}."""
    with open(sample_image_path, "rb") as img_file:
        detect_resp = client.post(
            "/api/detect",
            files={"file": (sample_image_path.name, img_file, "image/jpeg")},
        )
    assert detect_resp.status_code == 200
    annotated_url = detect_resp.json()["annotated_image"]

    # Retrieve annotated image
    img_resp = client.get(annotated_url)
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"].startswith("image/")
    assert len(img_resp.content) > 0


def test_result_image_not_found(client: TestClient):
    """Test GET /api/results/{filename} with non-existent file returns 404."""
    response = client.get("/api/results/non_existent_annotated_image_123.jpg")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_invalid_corrupted_image_handling(client: TestClient):
    """Test POST /api/detect with corrupted file content returns 400."""
    fake_corrupted_data = b"This is not a real image binary data"
    response = client.post(
        "/api/detect",
        files={"file": ("corrupt.jpg", fake_corrupted_data, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "invalid or corrupted" in response.json()["detail"].lower()


def test_unsupported_file_extension(client: TestClient):
    """Test POST /api/detect with unsupported file extension returns 400."""
    text_data = b"Some random text payload"
    response = client.post(
        "/api/detect",
        files={"file": ("readme.txt", text_data, "text/plain")},
    )
    assert response.status_code == 400
    assert "unsupported image extension" in response.json()["detail"].lower()


def test_empty_file_upload(client: TestClient):
    """Test POST /api/detect with empty byte stream returns 400."""
    response = client.post(
        "/api/detect",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_api_does_not_modify_best_pt(client: TestClient, sample_image_path: Path):
    """Verify that model best.pt SHA-256 checksum remains strictly unaltered after API calls."""
    best_pt_path = PROJECT_ROOT / "models" / "best.pt"
    if not best_pt_path.exists():
        best_pt_path = PROJECT_ROOT / "dataset" / "models" / "best.pt"
    assert best_pt_path.exists()

    # Pre-check hash
    pre_hash = get_file_sha256(best_pt_path)
    assert pre_hash == EXPECTED_BEST_PT_SHA256

    # Execute multiple API requests
    _ = client.get("/api/health")
    with open(sample_image_path, "rb") as img_file:
        _ = client.post(
            "/api/detect",
            files={"file": (sample_image_path.name, img_file, "image/jpeg")},
        )

    # Post-check hash
    post_hash = get_file_sha256(best_pt_path)
    assert post_hash == EXPECTED_BEST_PT_SHA256
    assert post_hash == pre_hash
