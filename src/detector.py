"""Reusable vehicle damage detector module using YOLOv8.

Independent of UI and downstream task logic.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union

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


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def get_default_project_root() -> Path:
    """Returns the project root directory dynamically without hardcoded paths."""
    return Path(__file__).resolve().parent.parent


@dataclass
class Detection:
    """Represents a single detected damage bounding box and its metadata."""
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    bounding_box_area: float
    image_width: int
    image_height: int
    normalized_box_area: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection object to dictionary."""
        return asdict(self)


@dataclass
class DetectionResult:
    """Structured container for prediction output on a single image."""
    image_path: str
    image_width: int
    image_height: int
    detections: List[Detection]
    device: str
    inference_time_ms: float = 0.0

    @property
    def count(self) -> int:
        """Returns the number of detections."""
        return len(self.detections)

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self):
        return iter(self.detections)

    def __getitem__(self, index: int) -> Detection:
        return self.detections[index]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full result to dictionary."""
        return {
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "count": self.count,
            "device": self.device,
            "inference_time_ms": self.inference_time_ms,
            "detections": [d.to_dict() for d in self.detections],
        }


class VehicleDamageDetector:
    """Reusable vehicle damage detector loading trained YOLOv8 model in memory."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize detector and load model into memory.

        Args:
            model_path: Path to best.pt. If None, checks settings.MODEL_PATH, then dataset/models/best.pt then models/best.pt.
            confidence_threshold: Minimum confidence score (0.0 to 1.0). If None, uses settings.CONFIDENCE_THRESHOLD.
            iou_threshold: Non-maximum suppression IoU threshold. If None, uses settings.IOU_THRESHOLD.
            device: 'cuda', 'cpu', or None to automatically detect CUDA.
        """
        self.confidence_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else float(settings.CONFIDENCE_THRESHOLD)
        )
        self.iou_threshold = (
            float(iou_threshold)
            if iou_threshold is not None
            else float(settings.IOU_THRESHOLD)
        )
        self.root_dir = get_default_project_root()

        # 1. Resolve model path
        self.model_path = self._resolve_model_path(model_path)

        # 2. Select execution device
        self.device = self._select_device(device)

        # 3. Load model into memory once
        self.model = self._load_model()
        self.class_names: Dict[int, str] = self.model.names

    def _resolve_model_path(self, model_path: Optional[Union[str, Path]]) -> Path:
        """Resolves model path dynamically."""
        if model_path is not None:
            resolved = Path(model_path)
            if not resolved.is_absolute():
                resolved = (self.root_dir / resolved).resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Specified model path does not exist: {resolved}")
            return resolved

        # Check configured settings.MODEL_PATH
        if settings.MODEL_PATH is not None:
            config_resolved = Path(settings.MODEL_PATH)
            if not config_resolved.is_absolute():
                config_resolved = (self.root_dir / config_resolved).resolve()
            if config_resolved.exists():
                return config_resolved

        # Default search order: dataset/models/best.pt, then models/best.pt
        primary = (self.root_dir / "dataset" / "models" / "best.pt").resolve()
        fallback = (self.root_dir / "models" / "best.pt").resolve()

        if primary.exists():
            return primary
        elif fallback.exists():
            return fallback
        else:
            raise FileNotFoundError(
                f"Model best.pt not found at default locations:\n  - {primary}\n  - {fallback}"
            )

    def _select_device(self, requested_device: Optional[str]) -> str:
        """Determines computation device (CUDA/GPU or CPU)."""
        if requested_device is not None:
            clean_dev = requested_device.strip().lower()
            if clean_dev.startswith("cuda") and not torch.cuda.is_available():
                print(f"[Warning] CUDA requested ('{requested_device}') but not available. Falling back to CPU.")
                return "cpu"
            return clean_dev

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> YOLO:
        """Loads YOLOv8 model weights in memory."""
        try:
            model = YOLO(str(self.model_path))
            return model
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model from '{self.model_path}': {exc}") from exc

    def predict(self, image_path: Union[str, Path]) -> DetectionResult:
        """Runs vehicle damage detection on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            DetectionResult containing structured detection boxes and metadata.

        Raises:
            FileNotFoundError: If image file does not exist.
            ValueError: If file is not a supported image format.
        """
        path = Path(image_path)
        if not path.is_absolute():
            path = (self.root_dir / path).resolve()
        else:
            path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"Image not found at path: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension '{path.suffix}'. Expected one of {SUPPORTED_EXTENSIONS}"
            )

        start_time = time.perf_counter()

        # Run inference using the pre-loaded in-memory model
        results = self.model.predict(
            source=str(path),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        if not results:
            return DetectionResult(
                image_path=str(path),
                image_width=0,
                image_height=0,
                detections=[],
                device=self.device,
                inference_time_ms=round(inference_time_ms, 2),
            )

        res = results[0]
        # orig_shape is (height, width)
        image_height, image_width = int(res.orig_shape[0]), int(res.orig_shape[1])
        total_image_area = float(image_width * image_height)

        detections: List[Detection] = []
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.class_names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0].item())

                xyxy = box.xyxy[0].tolist()
                x1 = float(xyxy[0])
                y1 = float(xyxy[1])
                x2 = float(xyxy[2])
                y2 = float(xyxy[3])

                box_width = max(0.0, x2 - x1)
                box_height = max(0.0, y2 - y1)
                box_area = box_width * box_height
                norm_area = box_area / total_image_area if total_image_area > 0 else 0.0

                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=confidence,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        bounding_box_area=box_area,
                        image_width=image_width,
                        image_height=image_height,
                        normalized_box_area=norm_area,
                    )
                )

        return DetectionResult(
            image_path=str(path),
            image_width=image_width,
            image_height=image_height,
            detections=detections,
            device=self.device,
            inference_time_ms=round(inference_time_ms, 2),
        )
