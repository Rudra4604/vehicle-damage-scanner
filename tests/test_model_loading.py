"""Test suite for YOLOv8 model loading and basic inference.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
from ultralytics import YOLO

from src.model_loader import get_project_root, locate_model, load_model, run_sample_inference, detect_device



EXPECTED_CLASSES = {'dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat'}


def test_model_file_exists():
    """Verify that best.pt exists and has valid size."""
    root = get_project_root()
    # Check both locations if present
    primary_path = root / "dataset" / "models" / "best.pt"
    models_path = root / "models" / "best.pt"
    
    assert primary_path.exists() or models_path.exists(), "best.pt not found in dataset/models/ or models/"
    target_path = primary_path if primary_path.exists() else models_path
    size = target_path.stat().st_size
    print(f"\n[Test 1] Found model at: {target_path} (Size: {size:,} bytes)")
    assert size > 1_000_000, "Model file seems too small or corrupt"


def test_model_load_and_exact_classes():
    """Verify the model loads cleanly and reports exactly the six expected classes."""
    root = get_project_root()
    model_path = root / "dataset" / "models" / "best.pt"
    if not model_path.exists():
        model_path = root / "models" / "best.pt"

    print(f"\n[Test 2] Loading model from: {model_path}")
    model = YOLO(str(model_path))
    assert model is not None, "Failed to instantiate YOLO model"
    assert hasattr(model, "names"), "Loaded model must have 'names' attribute"

    actual_classes = set(model.names.values()) if isinstance(model.names, dict) else set(model.names)
    print(f"[Test 2] Detected classes: {actual_classes}")
    assert actual_classes == EXPECTED_CLASSES, (
        f"Classes mismatch! Expected {EXPECTED_CLASSES}, but got {actual_classes}"
    )


def test_cuda_detection():
    """Detect whether CUDA/GPU is available."""
    device = detect_device()
    print(f"\n[Test 3] Execution device detected: {device.upper()}")
    assert device in ["cuda", "cpu"]


def test_sample_inference():
    """Verify inference runs on a test image from dataset/test/images."""
    root = get_project_root()
    model_path = root / "dataset" / "models" / "best.pt"
    if not model_path.exists():
        model_path = root / "models" / "best.pt"

    model = YOLO(str(model_path))
    test_images_dir = root / "tests" / "fixtures" if (root / "tests" / "fixtures").exists() else root / "dataset" / "test" / "images"
    assert test_images_dir.exists(), f"Test images directory does not exist: {test_images_dir}"
    
    results = run_sample_inference(model, test_images_dir)
    assert results is not None, "Inference failed on test dataset image"
    print("\n[Test 4] Inference test passed successfully.")


if __name__ == "__main__":
    print("=" * 60)
    print(" Running Stage 1 Verification Suite")
    print("=" * 60)
    test_model_file_exists()
    test_cuda_detection()
    test_model_load_and_exact_classes()
    test_sample_inference()
    print("\n" + "=" * 60)
    print(" ALL STAGE 1 VERIFICATIONS PASSED SUCCESSFULLY")
    print("=" * 60)

