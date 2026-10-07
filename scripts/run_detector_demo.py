"""Demo script for running VehicleDamageDetector on real test images.
"""
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import VehicleDamageDetector


def main():
    detector = VehicleDamageDetector()

    test_images_dir = PROJECT_ROOT / "dataset" / "test" / "images"
    if not test_images_dir.exists():
        print(f"Test directory not found: {test_images_dir}")
        return

    # Select the first sample image (or 000012.jpg if present)
    target_name = "000012.jpg"
    target_path = test_images_dir / target_name
    if not target_path.exists():
        images = [f for f in test_images_dir.iterdir() if f.suffix.lower() in {".jpg", ".png"}]
        if not images:
            print("No test images available.")
            return
        target_path = images[0]

    result = detector.predict(target_path)

    print("=" * 50)
    print(f"Device: {result.device.upper()}")
    print(f"Image: {Path(result.image_path).name}")
    print(f"Resolution: {result.image_width}x{result.image_height}")
    print(f"Inference Time: {result.inference_time_ms:.1f}ms")
    print(f"Detections: {result.count}")
    print("=" * 50)

    if result.count == 0:
        print("No damage detected above threshold.")
    else:
        for idx, det in enumerate(result.detections, 1):
            box = [round(det.x1, 1), round(det.y1, 1), round(det.x2, 1), round(det.y2, 1)]
            print(f"\n{idx}. {det.class_name}")
            print(f"   confidence: {det.confidence:.3f}")
            print(f"   box: {box}")
            print(f"   box area: {det.bounding_box_area:,.1f} px² ({det.normalized_box_area:.2%} of image)")


if __name__ == "__main__":
    main()
