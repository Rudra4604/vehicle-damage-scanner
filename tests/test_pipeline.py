"""Unit tests for the VehicleDamagePipeline module.
"""
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pytest
from src.detector import VehicleDamageDetector, Detection, DetectionResult
from src.pipeline import VehicleDamagePipeline, PipelineResult


@pytest.fixture(scope="session")
def pipeline() -> VehicleDamagePipeline:
    """Fixture providing a reusable VehicleDamagePipeline instance."""
    return VehicleDamagePipeline()


@pytest.fixture(scope="session")
def sample_test_image() -> Path:
    """Image with known damage (000012.jpg has flat tire)."""
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "000012.jpg"
    if fixture_path.exists():
        return fixture_path
    img_path = PROJECT_ROOT / "dataset" / "test" / "images" / "000012.jpg"
    assert img_path.exists(), f"Sample test image missing: {img_path}"
    return img_path


def test_pipeline_initialization(pipeline: VehicleDamagePipeline):
    """Verify pipeline initializes with underlying detector and default output directory."""
    assert pipeline.detector is not None
    assert isinstance(pipeline.detector, VehicleDamageDetector)
    assert pipeline.output_dir.exists()


def test_successful_inference(pipeline: VehicleDamagePipeline, sample_test_image: Path):
    """Test successful vehicle damage inference returning PipelineResult."""
    result = pipeline.analyze_image(sample_test_image)

    assert isinstance(result, PipelineResult)
    assert result.has_damage is True
    assert result.damage_status == "Damage Detected"
    assert result.total_detections > 0
    assert len(result.detections) == result.total_detections
    assert result.device in ["cpu", "cuda"]
    assert result.inference_time_ms > 0


def test_output_structure(pipeline: VehicleDamagePipeline, sample_test_image: Path):
    """Test the complete output structure and dictionary serialization."""
    result = pipeline.analyze_image(sample_test_image)

    # Check PipelineResult fields
    assert hasattr(result, "image_path")
    assert hasattr(result, "image_width")
    assert hasattr(result, "image_height")
    assert hasattr(result, "damage_status")
    assert hasattr(result, "has_damage")
    assert hasattr(result, "total_detections")
    assert hasattr(result, "unique_categories")
    assert hasattr(result, "average_confidence")
    assert hasattr(result, "highest_confidence_detection")
    assert hasattr(result, "total_damaged_area")
    assert hasattr(result, "damage_area_percentage")
    assert hasattr(result, "detections")
    assert hasattr(result, "annotated_image_path")
    assert hasattr(result, "device")
    assert hasattr(result, "inference_time_ms")

    # Check to_dict() serialization for web consumption
    data = result.to_dict()
    assert isinstance(data, dict)
    expected_keys = {
        "image_path", "image_width", "image_height", "damage_status", "has_damage",
        "total_detections", "unique_categories", "average_confidence",
        "highest_confidence_detection", "total_damaged_area", "damage_area_percentage",
        "annotated_image_path", "device", "inference_time_ms", "detections",
    }
    assert expected_keys.issubset(data.keys())
    assert isinstance(data["detections"], list)
    assert data["total_detections"] == len(data["detections"])
    if data["highest_confidence_detection"] is not None:
        assert isinstance(data["highest_confidence_detection"], dict)
        assert "class_name" in data["highest_confidence_detection"]
        assert "confidence" in data["highest_confidence_detection"]


def test_annotated_image_creation(pipeline: VehicleDamagePipeline, sample_test_image: Path, tmp_path: Path):
    """Test generating and saving an annotated image with detection bounding boxes."""
    custom_output = tmp_path / "custom_annotated.jpg"
    result = pipeline.analyze_image(
        sample_test_image,
        annotate=True,
        annotated_output_path=custom_output,
    )

    assert result.annotated_image_path is not None
    output_file = Path(result.annotated_image_path)
    assert output_file.exists()
    assert output_file.stat().st_size > 0

    # Verify that the generated image is a valid OpenCV image matching input dimensions
    img = cv2.imread(str(output_file))
    assert img is not None
    assert img.shape[0] == result.image_height
    assert img.shape[1] == result.image_width


def test_zero_detection_handling(pipeline: VehicleDamagePipeline, sample_test_image: Path):
    """Test handling when no damage meets confidence threshold."""
    strict_detector = VehicleDamageDetector(confidence_threshold=0.999)
    strict_pipeline = VehicleDamagePipeline(detector=strict_detector)

    result = strict_pipeline.analyze_image(sample_test_image, annotate=True)
    assert isinstance(result, PipelineResult)

    if result.total_detections == 0:
        assert result.damage_status == "No Damage Detected"
        assert result.has_damage is False
        assert result.total_detections == 0
        assert result.unique_categories == []
        assert result.average_confidence == 0.0
        assert result.highest_confidence_detection is None
        assert result.total_damaged_area == 0.0
        assert result.damage_area_percentage == 0.0
        assert result.detections == []
        # Annotated image should still be created even with 0 detections
        assert result.annotated_image_path is not None
        assert Path(result.annotated_image_path).exists()


def test_invalid_image_path(pipeline: VehicleDamagePipeline):
    """Test that missing image files raise FileNotFoundError."""
    invalid_path = PROJECT_ROOT / "dataset" / "test" / "images" / "non_existent_file_98765.jpg"
    with pytest.raises(FileNotFoundError):
        pipeline.analyze_image(invalid_path)


def test_unsupported_image_format(pipeline: VehicleDamagePipeline, tmp_path: Path):
    """Test that unsupported image extensions raise ValueError."""
    fake_txt = tmp_path / "vehicle.txt"
    fake_txt.write_text("not an image")
    with pytest.raises(ValueError):
        pipeline.analyze_image(fake_txt)


def test_multiple_damage_detections_handling(tmp_path: Path):
    """Test pipeline handling and aggregation with multiple damage detections."""
    class MockDetector:
        def __init__(self):
            self.device = "cpu"
            self.model_path = PROJECT_ROOT / "dataset" / "models" / "best.pt"

        def predict(self, image_path):
            return DetectionResult(
                image_path=str(image_path),
                image_width=1000,
                image_height=800,
                device="cpu",
                inference_time_ms=15.0,
                detections=[
                    Detection(
                        class_id=0,
                        class_name="dent",
                        confidence=0.85,
                        x1=50.0,
                        y1=50.0,
                        x2=150.0,
                        y2=150.0,
                        bounding_box_area=10000.0,
                        image_width=1000,
                        image_height=800,
                        normalized_box_area=0.0125,
                    ),
                    Detection(
                        class_id=1,
                        class_name="scratch",
                        confidence=0.75,
                        x1=200.0,
                        y1=200.0,
                        x2=400.0,
                        y2=300.0,
                        bounding_box_area=20000.0,
                        image_width=1000,
                        image_height=800,
                        normalized_box_area=0.025,
                    ),
                ],
            )

    mock_detector = MockDetector()
    mock_pipeline = VehicleDamagePipeline(detector=mock_detector)

    # Use actual test image so cv2 annotation can render
    fixture_img = PROJECT_ROOT / "tests" / "fixtures" / "000012.jpg"
    actual_img = fixture_img if fixture_img.exists() else PROJECT_ROOT / "dataset" / "test" / "images" / "000012.jpg"
    out_img = tmp_path / "mock_annotated.jpg"
    result = mock_pipeline.analyze_image(actual_img, annotate=True, annotated_output_path=out_img)

    assert result.has_damage is True
    assert result.damage_status == "Damage Detected"
    assert result.total_detections == 2
    assert "dent" in result.unique_categories
    assert "scratch" in result.unique_categories
    assert pytest.approx(result.average_confidence, 0.001) == 0.80
    assert result.highest_confidence_detection is not None
    assert result.highest_confidence_detection.class_name == "dent"
    assert result.highest_confidence_detection.confidence == 0.85
    assert result.total_damaged_area == 30000.0
    assert out_img.exists()
