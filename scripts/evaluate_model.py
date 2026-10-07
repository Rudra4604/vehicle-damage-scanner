"""Standalone script to run formal test set evaluation of the trained YOLOv8n detector.
"""
import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 detector on 374-image test set.")
    parser.add_argument("--model", type=str, default=None, help="Path to best.pt (default auto-detected)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset folder (default auto-detected)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default evaluation/detection)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda')")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        model_path=args.model,
        dataset_dir=args.dataset,
        output_dir=args.output,
        device=args.device,
    )

    results = evaluator.run_evaluation(
        imgsz=args.imgsz,
        batch_size=args.batch,
        split="test",
    )

    print("\n" + "=" * 65)
    print("                     EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Images Evaluated : {results.num_images}")
    print(f"Ground Truth BBoxes: {results.total_instances}")
    print(f"Device Used       : {results.device.upper()}")
    print("-" * 65)
    print(f"Overall Precision : {results.overall_precision:.4f}  (Colab benchmark: 0.753)")
    print(f"Overall Recall    : {results.overall_recall:.4f}  (Colab benchmark: 0.686)")
    print(f"Overall mAP@0.5   : {results.overall_map50:.4f}  (Colab benchmark: 0.7219)")
    print(f"Overall mAP@50-95 : {results.overall_map50_95:.4f}  (Colab benchmark: 0.5602)")
    print(f"Harmonic F1 Score : {results.overall_f1_harmonic:.4f}")
    print(f"Macro-average F1  : {results.overall_f1_macro:.4f}")
    print("-" * 65)
    print("Per-Class Results:")
    print(f"{'Class':<15} {'Instances':<10} {'P':<8} {'R':<8} {'F1':<8} {'mAP50':<8} {'mAP50-95':<8}")
    for c in results.per_class:
        print(f"{c.class_name:<15} {c.instances:<10} {c.precision:<8.3f} {c.recall:<8.3f} {c.f1:<8.3f} {c.map50:<8.3f} {c.map50_95:<8.3f}")
    print("=" * 65)
    print(f"\nArtifacts generated in:\n  {evaluator.output_dir}")
    print("  - results.json")
    print("  - results.csv")
    print("  - report.md")
    print("  - plots/ (PR_curve.png, F1_curve.png, confusion_matrix.png, per_class_metrics.png, etc.)")


if __name__ == "__main__":
    main()
