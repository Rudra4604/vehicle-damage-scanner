"""Demonstration script for running the VehicleDamagePipeline on a real test image.
"""
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import VehicleDamagePipeline


def main():
    pipeline = VehicleDamagePipeline()

    test_images_dir = PROJECT_ROOT / "dataset" / "test" / "images"
    target_img = test_images_dir / "000012.jpg"
    if not target_img.exists():
        candidates = list(test_images_dir.glob("*.jpg"))
        if not candidates:
            print("No test images found.")
            return
        target_img = candidates[0]

    # Run analysis with annotation enabled
    result = pipeline.analyze_image(target_img, annotate=True)

    print("=" * 50)
    print("VEHICLE DAMAGE ANALYSIS")
    print("=" * 50)
    print(f"Image: {Path(result.image_path).name}")
    print(f"Resolution: {result.image_width} x {result.image_height}")
    print(f"Device: {result.device.upper()} ({result.inference_time_ms:.1f}ms)")
    print()
    print(f"Damage Status: {result.damage_status}")
    print(f"Total Detections: {result.total_detections}")
    print(f"Damage Types: {', '.join(result.unique_categories) if result.unique_categories else 'None'}")
    print(f"Average Confidence: {result.average_confidence:.2%}")
    print(f"Damaged Area: {result.total_damaged_area:,.1f} px² ({result.damage_area_percentage:.2f}% of vehicle image)")
    print()
    print("Detections:")
    if result.total_detections == 0:
        print("  None")
    else:
        for idx, det in enumerate(result.detections, 1):
            box = [round(det.x1, 1), round(det.y1, 1), round(det.x2, 1), round(det.y2, 1)]
            print(f"{idx}. {det.class_name}")
            print(f"   Confidence: {det.confidence:.2%}")
            print(f"   Box: {box}")

    print()
    print("Annotated image saved to:")
    print(f"{result.annotated_image_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
