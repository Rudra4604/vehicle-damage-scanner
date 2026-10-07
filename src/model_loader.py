from pathlib import Path
import sys
from typing import Optional, Union
import torch
from ultralytics import YOLO

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import settings
except ModuleNotFoundError:
    from config import settings


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parent.parent


def detect_device() -> str:
    """Detects whether CUDA/GPU or CPU is available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        print(f"[Device] CUDA/GPU is available: {gpu_name} ({device_count} device(s))")
        return "cuda"
    else:
        print("[Device] CUDA/GPU not available. Using CPU.")
        return "cpu"


def locate_model(model_rel_path: Optional[Union[str, Path]] = None) -> Path:
    """Locates the trained model file."""
    root = get_project_root()
    if model_rel_path is not None:
        target = Path(model_rel_path)
    else:
        target = Path(settings.MODEL_PATH)

    model_path = root / target if not target.is_absolute() else target

    if not model_path.exists():
        # Fallback check under models/best.pt then dataset/models/best.pt if needed
        primary_fallback = root / "models" / "best.pt"
        dataset_fallback = root / "dataset" / "models" / "best.pt"
        if primary_fallback.exists():
            return primary_fallback
        elif dataset_fallback.exists():
            print(f"[Warning] Model not found at '{model_path}', found at fallback '{dataset_fallback}'.")
            return dataset_fallback
        raise FileNotFoundError(f"Trained model not found at '{model_path}' or fallback locations.")

    return model_path


def load_model(model_path: Path = None):
    """Loads the YOLO model and prints class names."""
    if model_path is None:
        model_path = locate_model()

    print(f"\n[1/3] Locating model at: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    print("[2/3] Loading YOLO model via Ultralytics...")
    model = YOLO(str(model_path))
    print("[SUCCESS] Model loaded successfully!")

    # Print class names
    class_names = model.names
    print(f"\n[Classes] Detected {len(class_names)} classes:")
    if isinstance(class_names, dict):
        for idx, name in class_names.items():
            print(f"  - Class {idx}: {name}")
    elif isinstance(class_names, list):
        for idx, name in enumerate(class_names):
            print(f"  - Class {idx}: {name}")

    return model


def run_sample_inference(model, test_images_dir: Path = None):
    """Runs one sample inference on an image from dataset/test/images."""
    root = get_project_root()
    if test_images_dir is None:
        test_images_dir = root / "dataset" / "test" / "images"

    print(f"\n[3/3] Checking test images in: {test_images_dir}")
    if not test_images_dir.exists() or not test_images_dir.is_dir():
        print(f"[Info] Test images directory does not exist: {test_images_dir}")
        return None

    # Search for an image file
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    sample_image = None
    for file in sorted(test_images_dir.iterdir()):
        if file.suffix.lower() in valid_extensions:
            sample_image = file
            break

    if sample_image is None:
        print("[Info] No test images found in directory.")
        return None

    print(f"[Inference] Running sample inference on: {sample_image.name}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = model.predict(source=str(sample_image), device=device, verbose=False)

    for i, res in enumerate(results):
        num_boxes = len(res.boxes)
        print(f"[Result] Detected {num_boxes} object(s) in {sample_image.name}:")
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, str(cls_id)) if isinstance(model.names, dict) else model.names[cls_id]
            conf = float(box.conf[0].item())
            coords = [round(x, 1) for x in box.xyxy[0].tolist()]
            print(f"  - Class: '{cls_name}' | Confidence: {conf:.2%} | Box: {coords}")

    return results


def main():
    print("=" * 60)
    print("      DLL Project - Model Loading & Verification")
    print("=" * 60)

    # 1. Device check
    device = detect_device()

    # 2. Locate & load model
    model_path = locate_model()
    model = load_model(model_path)

    # 3. Sample inference
    run_sample_inference(model)

    print("\n" + "=" * 60)
    print("[COMPLETED] Model verification finished successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
