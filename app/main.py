"""FastAPI REST backend for vehicle damage detection.
"""
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Optional
import uuid

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.detector import SUPPORTED_EXTENSIONS, get_default_project_root
from src.pipeline import VehicleDamagePipeline, PipelineResult

logger = logging.getLogger("vehicle_damage_api")
logging.basicConfig(level=logging.INFO)

# Global pipeline instance initialized on startup
pipeline_instance: Optional[VehicleDamagePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load pipeline and detector once into memory."""
    global pipeline_instance
    logger.info("Initializing VehicleDamagePipeline and loading model weights into memory...")
    pipeline_instance = VehicleDamagePipeline()
    logger.info(f"Pipeline ready. Running on device: {pipeline_instance.detector.device.upper()}")
    # Safe output pruning on startup
    try:
        pruned_count = pipeline_instance.prune_outputs()
        if pruned_count > 0:
            logger.info(f"Startup maintenance: pruned {pruned_count} old/excess output files.")
    except Exception as exc:
        logger.warning(f"Startup output maintenance failed: {exc}")
    yield
    logger.info("Shutting down VehicleDamagePipeline service.")


def get_pipeline() -> VehicleDamagePipeline:
    """Retrieve or lazily initialize the singleton pipeline."""
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = VehicleDamagePipeline()
    return pipeline_instance


app = FastAPI(
    title="Vehicle Damage Detection API",
    description="REST backend for automated vehicle damage identification, bounding box extraction, and annotated visualization rendering.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS using centralized settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", summary="Health Check")
async def health_check() -> Dict[str, Any]:
    """Returns the operational status of the service and active compute device."""
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "service": "vehicle-damage-detection",
        "version": "1.0.0",
        "device": pipeline.detector.device,
        "classes": list(pipeline.detector.class_names.values()),
    }


@app.post("/api/detect", summary="Detect Vehicle Damage")
async def detect_damage(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accepts an uploaded vehicle image, executes damage detection, and returns structured results."""
    # 1. Validate file presence and filename
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in upload request.",
        )

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image extension '{file_ext}'. Allowed formats: {sorted(list(SUPPORTED_EXTENSIONS))}",
        )

    # 2. Read file contents and check for empty upload
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(exc)}",
        )

    if not content or len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # 3. Create a secure temporary file on disk for processing
    pipeline = get_pipeline()
    temp_dir = tempfile.gettempdir()
    unique_name = f"upload_{uuid.uuid4().hex[:12]}{file_ext}"
    temp_path = Path(temp_dir) / unique_name

    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        # 4. Verify image integrity with OpenCV
        test_read = cv2.imread(str(temp_path))
        if test_read is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or corrupted image file. OpenCV could not decode the image data.",
            )

        # 5. Run inference through VehicleDamagePipeline
        annotated_filename = f"annotated_{uuid.uuid4().hex[:12]}{file_ext}"
        annotated_dest = pipeline.output_dir / annotated_filename

        try:
            result: PipelineResult = pipeline.analyze_image(
                image_path=temp_path,
                annotate=True,
                annotated_output_path=annotated_dest,
            )
        except Exception as exc:
            logger.error(f"Pipeline inference failed: {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference processing error: {str(exc)}",
            )

        # 6. Build structured serializable JSON response
        result_dict = result.to_dict()
        annotated_url = f"/api/results/{annotated_filename}"

        return {
            "damage_status": result_dict["damage_status"],
            "has_damage": result_dict["has_damage"],
            "total_detections": result_dict["total_detections"],
            "unique_categories": result_dict["unique_categories"],
            "average_confidence": result_dict["average_confidence"],
            "damage_area_percentage": result_dict["damage_area_percentage"],
            "detections": result_dict["detections"],
            "image_width": result_dict["image_width"],
            "image_height": result_dict["image_height"],
            "inference_time_ms": result_dict["inference_time_ms"],
            "annotated_image": annotated_url,
            "annotated_image_path": str(annotated_dest),
        }

    finally:
        # 7. Safe temporary file cleanup
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as cleanup_err:
                logger.warning(f"Could not remove temporary file '{temp_path}': {cleanup_err}")


@app.get("/api/results/{filename}", summary="Retrieve Annotated Result Image")
async def get_result_image(filename: str):
    """Serves the generated annotated image file from the outputs directory."""
    # Prevent path traversal attacks
    safe_name = Path(filename).name
    pipeline = get_pipeline()
    file_path = (pipeline.output_dir / safe_name).resolve()

    # Ensure path resides strictly within the outputs directory
    try:
        file_path.relative_to(pipeline.output_dir.resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename requested.",
        )

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotated result image '{safe_name}' not found.",
        )

    # Determine media type from suffix
    ext = file_path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    media_type = media_type_map.get(ext, "application/octet-stream")

    return FileResponse(path=str(file_path), media_type=media_type, filename=safe_name)


# Mount static frontend application (React 19 SPA as primary, vanilla SPA as fallback)
FRONTEND_DIR = get_default_project_root() / "app" / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"

if FRONTEND_DIST_DIR.exists() and (FRONTEND_DIST_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="frontend")
elif FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

