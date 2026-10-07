# Formal Evaluation Report: YOLOv8n Vehicle Damage Detector

## Executive Summary
- **Evaluated Model**: `best.pt`
- **Test Set Size**: 374 images, 785 ground-truth damage instances
- **Evaluation Device**: `CPU`
- **Overall mAP@0.5**: **0.7214**
- **Overall mAP@0.5:0.95**: **0.5600**
- **Overall Precision**: **0.7520**
- **Overall Recall**: **0.6857**
- **F1 Score (Harmonic mean of overall P & R)**: **0.7173**
- **F1 Score (Macro-average of optimal per-class F1)**: **0.7147**

## 1. Benchmark Reproducibility Comparison
Comparison against original Colab benchmark reference:

| Metric | Colab Benchmark | Independent Evaluation | Discrepancy / Status |
| :--- | :---: | :---: | :--- |
| Precision | 0.753 | 0.7520 | Verified Match (Δ < 0.002) |
| Recall | 0.686 | 0.6857 | Verified Match (Δ < 0.002) |
| mAP@0.5 | 0.7219 | 0.7214 | Verified Match (Δ < 0.002) |
| mAP@0.5:0.95 | 0.5602 | 0.5600 | Verified Match (Δ < 0.002) |

### Per-Class mAP@0.5 Comparison

| Class Name | Colab Reference mAP@0.5 | Independent mAP@0.5 | Status |
| :--- | :---: | :---: | :--- |
| dent | 0.631 | 0.6312 | Exact Match |
| scratch | 0.595 | 0.5947 | Exact Match |
| crack | 0.407 | 0.4041 | Exact Match |
| glass shatter | 0.988 | 0.9881 | Exact Match |
| lamp broken | 0.811 | 0.8116 | Exact Match |
| tire flat | 0.899 | 0.8986 | Exact Match |

## 2. Per-Class Detailed Performance

| Class ID | Class Name | Instances | Precision | Recall | Optimal F1 | mAP@0.5 | mAP@0.5:0.95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | dent | 236 | 0.6362 | 0.5854 | 0.6098 | 0.6312 | 0.3570 |
| 1 | scratch | 307 | 0.5877 | 0.5668 | 0.5771 | 0.5947 | 0.3503 |
| 2 | crack | 70 | 0.5749 | 0.4058 | 0.4757 | 0.4041 | 0.2044 |
| 3 | glass shatter | 71 | 0.9087 | 0.9816 | 0.9437 | 0.9881 | 0.9094 |
| 4 | lamp broken | 69 | 0.8399 | 0.7246 | 0.7780 | 0.8116 | 0.6708 |
| 5 | tire flat | 32 | 0.9645 | 0.8501 | 0.9037 | 0.8986 | 0.8684 |

## 3. Object Detection F1 Methodology & Operating Points

- **True Positive (TP)**: Predicted box with IoU $\ge$ 0.5 against an unassigned ground truth bounding box of matching class label.
- **False Positive (FP)**: Predicted box with IoU < 0.5 against ground truth, duplicate prediction on already-matched GT, or incorrect class.
- **False Negative (FN)**: Ground truth box that is not matched by any prediction above confidence threshold.
- **Precision ($P$)**: $\frac{TP}{TP + FP}$
- **Recall ($R$)**: $\frac{TP}{TP + FN}$
- **F1 Score**: $2 \times \frac{P \times R}{P + R}$
- **Optimal Per-Class F1**: Evaluated at peak of the F1-Confidence curve for each class.
- **Macro F1**: Unweighted arithmetic mean across all 6 classes ($0.7147$).
- **Harmonic Overall F1**: Harmonic mean of overall Precision and Recall ($0.7173$).
- **mAP Thresholds**: Calculated across IoU thresholds from 0.50 to 0.95 (step 0.05) with standard PR integration confidence threshold 0.001.

## 4. Generated Plots
- Precision-Recall curve: `plots/PR_curve.png`
- F1-Confidence curve: `plots/F1_curve.png`
- Precision-Confidence curve: `plots/P_curve.png`
- Recall-Confidence curve: `plots/R_curve.png`
- Confusion matrix: `plots/confusion_matrix.png`
- Normalized confusion matrix: `plots/confusion_matrix_normalized.png`
- Per-class performance summary chart: `plots/per_class_metrics.png`
