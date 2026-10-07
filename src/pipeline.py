"""End-to-end vehicle damage inference pipeline.

Wraps VehicleDamageDetector to produce aggregated analytics, damage status,
and annotated visualizations for downstream UI/API consumption.
"""
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import settings
    from src.detector import VehicleDamageDetector, Detection, DetectionResult, get_default_project_root
except ModuleNotFoundError:
    from config import settings
    from detector import VehicleDamageDetector, Detection, DetectionResult, get_default_project_root

logger = logging.getLogger("vehicle_damage_pipeline")


def prune_output_directory(
    output_dir: Optional[Union[str, Path]] = None,
    retention_hours: Optional[int] = None,
    max_files: Optional[int] = None,
) -> int:
    """Safely prunes old and excess generated output files from the output directory.

    Guarantees:
    - Never deletes models, dataset, source code, evaluation results, or project root.
    - Preserves .gitkeep placeholders.
    - Handles missing or empty directories gracefully.
    - Resilient to individual file deletion errors (will not crash).

    Args:
        output_dir: Target output directory. Defaults to settings.OUTPUT_DIR.
        retention_hours: Maximum file age in hours before removal. Defaults to settings.OUTPUT_RETENTION_HOURS.
        max_files: Maximum allowed output files. Defaults to settings.MAX_OUTPUT_FILES.

    Returns:
        The number of files successfully pruned.
    """
    root_dir = get_default_project_root().resolve()

    if output_dir is None:
        target_dir = (root_dir / settings.OUTPUT_DIR).resolve()
    else:
        target_dir = Path(output_dir)
        if not target_dir.is_absolute():
            target_dir = (root_dir / target_dir).resolve()
        else:
            target_dir = target_dir.resolve()

    # Rule 8: Handle missing or non-directory output paths safely
    if not target_dir.exists() or not target_dir.is_dir():
        return 0

    # Rules 3, 4, 5, 6, 7: Protect critical system and project directories
    protected_dirs = {
        root_dir,
        (root_dir / "models").resolve(),
        (root_dir / "dataset").resolve(),
        (root_dir / "src").resolve(),
        (root_dir / "evaluation").resolve(),
        (root_dir / "tests").resolve(),
        (root_dir / "app").resolve(),
        (root_dir / "scripts").resolve(),
    }
    # Also protect filesystem roots and parent directories of the workspace
    system_roots = {Path(p).resolve() for p in [Path.cwd().anchor, "/", "\\"] if Path(p).exists()}
    protected_dirs.update(system_roots)

    if target_dir in protected_dirs:
        logger.warning(f"Prune aborted: '{target_dir}' is a protected directory.")
        return 0

    if target_dir in root_dir.parents:
        logger.warning(f"Prune aborted: '{target_dir}' is an ancestor of project root.")
        return 0

    effective_retention = (
        int(retention_hours)
        if retention_hours is not None
        else int(settings.OUTPUT_RETENTION_HOURS)
    )
    effective_max = (
        int(max_files)
        if max_files is not None
        else int(settings.MAX_OUTPUT_FILES)
    )

    # Collect files in output directory (excluding .gitkeep and subdirectories)
    file_records: List[tuple[Path, float]] = []
    try:
        for entry in target_dir.iterdir():
            if entry.is_file() and entry.name != ".gitkeep":
                try:
                    stat_res = entry.stat()
                    file_records.append((entry, stat_res.st_mtime))
                except OSError:
                    continue
    except OSError as err:
        logger.warning(f"Failed to scan output directory for pruning: {err}")
        return 0

    pruned_count = 0
    now = time.time()
    cutoff_time = now - (effective_retention * 3600.0)

    surviving_files: List[tuple[Path, float]] = []

    # 1. Prune files older than retention hours
    for file_path, mtime in file_records:
        if mtime < cutoff_time:
            try:
                file_path.unlink()
                pruned_count += 1
            except OSError as err:
                logger.warning(f"Could not prune stale output file '{file_path}': {err}")
        else:
            surviving_files.append((file_path, mtime))

    # 2. Enforce MAX_OUTPUT_FILES (keep newest, prune oldest)
    if effective_max > 0 and len(surviving_files) > effective_max:
        surviving_files.sort(key=lambda item: item[1], reverse=True)
        excess = surviving_files[effective_max:]
        for file_path, _ in excess:
            try:
                file_path.unlink()
                pruned_count += 1
            except OSError as err:
                logger.warning(f"Could not prune excess output file '{file_path}': {err}")

    return pruned_count


# Distinct visually appealing BGR colors for each damage class
CLASS_COLORS = {
    0: (255, 140, 0),    # dent: Deep Sky Blue (BGR)
    1: (0, 165, 255),    # scratch: Orange
    2: (50, 50, 240),    # crack: Vivid Red
    3: (220, 20, 140),   # glass shatter: Violet/Pink
    4: (0, 215, 255),    # lamp broken: Gold / Bright Yellow
    5: (50, 205, 50),    # tire flat: Lime Green
}
DEFAULT_COLOR = (0, 255, 0)


@dataclass
class PipelineResult:
    """Structured high-level analysis result produced by VehicleDamagePipeline."""
    image_path: str
    image_width: int
    image_height: int
    damage_status: str               # "Damage Detected" or "No Damage Detected"
    has_damage: bool
    total_detections: int
    unique_categories: List[str]
    average_confidence: float
    highest_confidence_detection: Optional[Detection]
    total_damaged_area: float
    damage_area_percentage: float
    detections: List[Detection]
    annotated_image_path: Optional[str] = None
    device: str = "cpu"
    inference_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into JSON-compatible dictionary."""
        return {
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "damage_status": self.damage_status,
            "has_damage": self.has_damage,
            "total_detections": self.total_detections,
            "unique_categories": self.unique_categories,
            "average_confidence": round(self.average_confidence, 4),
            "highest_confidence_detection": (
                self.highest_confidence_detection.to_dict()
                if self.highest_confidence_detection is not None
                else None
            ),
            "total_damaged_area": round(self.total_damaged_area, 2),
            "damage_area_percentage": round(self.damage_area_percentage, 2),
            "annotated_image_path": self.annotated_image_path,
            "device": self.device,
            "inference_time_ms": self.inference_time_ms,
            "detections": [d.to_dict() for d in self.detections],
        }


class VehicleDamagePipeline:
    """High-level application inference pipeline for vehicle damage detection."""

    def __init__(
        self,
        detector: Optional[VehicleDamageDetector] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the pipeline, reusing an existing detector if provided."""
        self.root_dir = get_default_project_root()

        conf = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else float(settings.CONFIDENCE_THRESHOLD)
        )
        iou = (
            float(iou_threshold)
            if iou_threshold is not None
            else float(settings.IOU_THRESHOLD)
        )

        if detector is not None:
            self.detector = detector
        else:
            self.detector = VehicleDamageDetector(
                confidence_threshold=conf,
                iou_threshold=iou,
                device=device,
            )

        if output_dir is not None:
            self.output_dir = Path(output_dir)
            if not self.output_dir.is_absolute():
                self.output_dir = (self.root_dir / self.output_dir).resolve()
        else:
            cfg_out = settings.OUTPUT_DIR
            self.output_dir = (
                cfg_out.resolve()
                if cfg_out.is_absolute()
                else (self.root_dir / cfg_out).resolve()
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prune_outputs(
        self,
        retention_hours: Optional[int] = None,
        max_files: Optional[int] = None,
    ) -> int:
        """Prunes stale or excess output images in this pipeline's output directory."""
        return prune_output_directory(
            output_dir=self.output_dir,
            retention_hours=retention_hours,
            max_files=max_files,
        )

    def analyze_image(
        self,
        image_path: Union[str, Path],
        annotate: bool = True,
        annotated_output_path: Optional[Union[str, Path]] = None,
    ) -> PipelineResult:
        """Runs end-to-end inference and returns comprehensive analysis.

        Args:
            image_path: Path to the vehicle image.
            annotate: Whether to render and save an annotated image.
            annotated_output_path: Optional custom destination for the annotated image.

        Returns:
            PipelineResult containing summary analytics and detection boxes.
        """
        raw_result: DetectionResult = self.detector.predict(image_path)
        
        detections = raw_result.detections
        total_detections = len(detections)
        has_damage = total_detections > 0
        damage_status = "Damage Detected" if has_damage else "No Damage Detected"

        # Unique categories preserving original class order
        unique_categories: List[str] = []
        for d in detections:
            if d.class_name not in unique_categories:
                unique_categories.append(d.class_name)

        # Confidence statistics
        if total_detections > 0:
            confidences = [d.confidence for d in detections]
            avg_confidence = float(np.mean(confidences))
            highest_detection = max(detections, key=lambda d: d.confidence)
            total_damaged_area = float(sum(d.bounding_box_area for d in detections))
        else:
            avg_confidence = 0.0
            highest_detection = None
            total_damaged_area = 0.0

        image_area = float(raw_result.image_width * raw_result.image_height)
        damage_area_percentage = (
            (total_damaged_area / image_area * 100.0) if image_area > 0 else 0.0
        )

        pipeline_res = PipelineResult(
            image_path=raw_result.image_path,
            image_width=raw_result.image_width,
            image_height=raw_result.image_height,
            damage_status=damage_status,
            has_damage=has_damage,
            total_detections=total_detections,
            unique_categories=unique_categories,
            average_confidence=avg_confidence,
            highest_confidence_detection=highest_detection,
            total_damaged_area=total_damaged_area,
            damage_area_percentage=damage_area_percentage,
            detections=detections,
            annotated_image_path=None,
            device=raw_result.device,
            inference_time_ms=raw_result.inference_time_ms,
        )

        if annotate:
            saved_path = self.generate_annotated_image(
                pipeline_res,
                output_path=annotated_output_path,
            )
            pipeline_res.annotated_image_path = str(saved_path)

        return pipeline_res

    def generate_annotated_image(
        self,
        result: PipelineResult,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Renders bounding boxes, class labels, and confidence tags on the image.

        Args:
            result: PipelineResult object containing detections and image path.
            output_path: Path where annotated image should be saved.

        Returns:
            Path to the saved annotated image.
        """
        img_path = Path(result.image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Original image not found for annotation: {img_path}")

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image with OpenCV: {img_path}")

        # Draw each detection box and banner label
        for det in result.detections:
            color = CLASS_COLORS.get(det.class_id, DEFAULT_COLOR)
            x1, y1 = int(round(det.x1)), int(round(det.y1))
            x2, y2 = int(round(det.x2)), int(round(det.y2))

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)

            # Label text: class name + confidence
            label = f"{det.class_name} {det.confidence:.1%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Text background rectangle
            bg_y1 = max(0, y1 - text_h - 6)
            bg_y2 = y1
            bg_x2 = min(img.shape[1], x1 + text_w + 6)
            cv2.rectangle(img, (x1, bg_y1), (bg_x2, bg_y2), color, thickness=-1)

            # Text string (white text on colored background)
            cv2.putText(
                img,
                label,
                (x1 + 3, y1 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

        # Resolve output destination
        if output_path is not None:
            dest = Path(output_path)
            if not dest.is_absolute():
                dest = (self.root_dir / dest).resolve()
        else:
            dest = (self.output_dir / f"annotated_{img_path.name}").resolve()

        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), img)

        # Enforce output directory retention safely without crashing
        try:
            self.prune_outputs()
        except Exception as exc:
            logger.warning(f"Output pruning failed after saving image: {exc}")

        return dest
