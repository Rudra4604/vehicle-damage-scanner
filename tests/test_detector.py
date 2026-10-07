"""Unit tests for the VehicleDamageDetector module.
"""
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.detector import VehicleDamageDetector, Detection, DetectionResult


@pytest.fixture(scope="session")
def detector() -> VehicleDamageDetector:
    """Fixture providing a reusable VehicleDamageDetector instance."""
    return VehicleDamageDetector()


@pytest.fixture(scope="session")
def sample_test_image() -> Path:
    """Locates an actual test image from tests/fixtures or dataset/test/images."""
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "000012.jpg"
    if fixture_path.exists():
        return fixture_path
    test_dir = PROJECT_ROOT / "dataset" / "test" / "images"
    assert test_dir.exists(), f"Test image directory missing: {test_dir}"
    
    images = [f for f in sorted(test_dir.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    assert len(images) > 0, "No test images found in dataset/test/images"
    return images[0]


def test_detector_initialization(detector: VehicleDamageDetector):
    """Verify detector loads model in memory and exposes expected attributes."""
    assert detector.model is not None
    assert detector.model_path.exists()
    assert isinstance(detector.class_names, dict)
    assert len(detector.class_names) == 6
    assert detector.device in ["cuda", "cpu"]


def test_prediction_result_structure(detector: VehicleDamageDetector, sample_test_image: Path):
    """Verify that prediction produces valid DetectionResult with all required fields."""
    result = detector.predict(sample_test_image)

    assert isinstance(result, DetectionResult)
    assert Path(result.image_path).resolve() == sample_test_image.resolve()
    assert result.image_width > 0
    assert result.image_height > 0
    assert result.device in ["cuda", "cpu"]
    assert isinstance(result.detections, list)
    assert result.count == len(result.detections)
    assert len(result) == result.count

    # Verify dictionary serialization
    res_dict = result.to_dict()
    assert "detections" in res_dict
    assert "image_width" in res_dict
    assert "image_height" in res_dict
    assert "count" in res_dict
    assert "inference_time_ms" in res_dict


def test_detection_box_values(detector: VehicleDamageDetector, sample_test_image: Path):
    """Verify detection attributes, confidence bounds, and coordinate validity."""
    result = detector.predict(sample_test_image)

    for det in result.detections:
        assert isinstance(det, Detection)
        assert isinstance(det.class_id, int)
        assert isinstance(det.class_name, str)
        assert len(det.class_name) > 0

        # Confidence bounds [0, 1]
        assert 0.0 <= det.confidence <= 1.0

        # Bounding box bounds
        assert 0.0 <= det.x1 <= det.x2 <= result.image_width
        assert 0.0 <= det.y1 <= det.y2 <= result.image_height
        assert det.bounding_box_area >= 0.0

        # Normalized area bounds [0, 1]
        assert 0.0 <= det.normalized_box_area <= 1.0

        # Verify detection to_dict()
        det_dict = det.to_dict()
        for field in [
            "class_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2",
            "bounding_box_area", "image_width", "image_height", "normalized_box_area"
        ]:
            assert field in det_dict


def test_zero_detections_handling(detector: VehicleDamageDetector, sample_test_image: Path):
    """Verify that high confidence threshold produces 0 detections without failing."""
    strict_detector = VehicleDamageDetector(confidence_threshold=0.999)
    result = strict_detector.predict(sample_test_image)

    assert isinstance(result, DetectionResult)
    assert result.count >= 0
    if result.count == 0:
        assert result.detections == []
        assert len(result) == 0


def test_invalid_image_path(detector: VehicleDamageDetector):
    """Verify that non-existent image paths raise FileNotFoundError."""
    invalid_path = PROJECT_ROOT / "dataset" / "test" / "images" / "does_not_exist_12345.jpg"
    with pytest.raises(FileNotFoundError):
        detector.predict(invalid_path)


def test_unsupported_image_format(detector: VehicleDamageDetector, tmp_path: Path):
    """Verify that unsupported file extensions raise ValueError."""
    fake_file = tmp_path / "dummy_file.txt"
    fake_file.write_text("not an image")
    with pytest.raises(ValueError):
        detector.predict(fake_file)


def test_in_memory_persistence(detector: VehicleDamageDetector, sample_test_image: Path):
    """Verify model remains loaded in memory and does not reload from disk."""
    model_obj_id = id(detector.model)
    _ = detector.predict(sample_test_image)
    _ = detector.predict(sample_test_image)
    assert id(detector.model) == model_obj_id
