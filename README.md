# Vehicle Damage Scanner: Automated Car Damage Detection & Severity Assessment

An end-to-end computer vision and deep learning application for automated vehicular exterior damage detection, multi-class categorization, damage severity analytics, and interactive visual inspection.

The system utilizes an in-memory **YOLOv8 Nano (YOLOv8n)** model fine-tuned on the **CarDD (Car Damage Dataset)**, integrated with an asynchronous **FastAPI** backend and a responsive **React 19 / TypeScript** single-page web interface.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Supported Damage Classes](#supported-damage-classes)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Technology Stack](#technology-stack)
- [Model Information & Benchmark Metrics](#model-information--benchmark-metrics)
- [Backend & REST API](#backend--rest-api)
- [Frontend Interfaces](#frontend-interfaces)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Building & Developing the Frontend](#building--developing-the-frontend)
- [How to Use the Application](#how-to-use-the-application)
- [Testing Suite](#testing-suite)
- [Important Notes on Dataset & Model Weights](#important-notes-on-dataset--model-weights)

---

## Overview

Accurate and prompt vehicle damage assessment is essential for automotive insurance claims, car rental check-in inspections, and fleet management. Manual assessments are often subjective, slow, and prone to inconsistency.

**Vehicle Damage Scanner** solves this problem by providing an automated pipeline that:
1. Ingests exterior vehicle photographs.
2. Identifies and localizes damage instances with precise bounding boxes.
3. Classifies damage across 6 standard vehicular categories.
4. Calculates structural metrics including damaged surface area coverage (in $\text{px}^2$ and percentage of overall image) and detection confidence scores.
5. Returns both interactive visual overlays and structured JSON payloads for downstream claim processing.

---

## Key Features

- **Multi-Class Damage Localization:** Detects 6 distinct damage types using single-stage anchor-free object detection.
- **Severity & Surface Area Analytics:** Computes the total damaged bounding area, percentage of vehicle surface damaged, mean confidence, and highest-confidence detection.
- **Automated Visual Annotations:** Uses OpenCV to render anti-aliased, class-color-coded bounding boxes and labels onto output images.
- **High-Performance Asynchronous API:** Powered by FastAPI and Uvicorn, featuring multipart file validation, OpenCV decode checks, automatic temporary file cleanup, and automated OpenAPI documentation (`/docs`).
- **Interactive Dual Frontend:**
  - **Primary:** Modern React 19 + TypeScript Single-Page Application (SPA) built with Vite and Tailwind CSS.
  - **Fallback:** Standalone vanilla HTML5/ES6/Tailwind client requiring zero Node.js tooling.
- **Inspection Tools:** Drag-and-drop file upload, live scan sweep animation, 1.3x zoom toggle, full-screen modal viewer, and one-click JSON report export.
- **Reproducible Evaluation Suite:** Comprehensive evaluation pipeline matching Colab/Kaggle benchmarks across 374 test set images, complete with confusion matrices and PR curves.
- **Production-Ready Quality:** 39 automated unit and integration tests covering the API, detector, pipeline analytics, model loading, configuration, and frontend assets.

---

## Supported Damage Classes

The model is trained on the **CarDD (Car Damage Dataset)** and identifies 6 damage classes:

| Class ID | Class Name | Description & Typical Presentation | CarDD Total Instances | Test Set Instances |
| :---: | :--- | :--- | :---: | :---: |
| `0` | `dent` | Surface depressions and body panel deformations | 2,543 (29.1%) | 236 |
| `1` | `scratch` | Paint abrasions, clear-coat scuffs, and surface scrapes | 3,595 (41.1%) | 307 |
| `2` | `crack` | Fractures and split lines on bumpers, grilles, or body panels | 898 (10.3%) | 70 |
| `3` | `glass shatter` | Spider-web fractures and shattered window/windshield glass | 681 (7.8%) | 71 |
| `4` | `lamp broken` | Fractured or destroyed headlamps, tail lamps, and turn signals | 704 (8.1%) | 69 |
| `5` | `tire flat` | Deflated, punctured, or collapsed vehicle tires | 319 (3.6%) | 32 |
| **Total** | **6 Classes** | **CarDD Full Dataset Split** | **8,740 (100%)** | **785** |

---

## System Architecture & Workflow

```
                             +------------------------------------------+
                             |          User Web Browser / Client       |
                             +------------------------------------------+
                                                  |
                                   Upload Image (Drag & Drop / File)
                                                  v
                             +------------------------------------------+
                             |   React 19 SPA (app/frontend/dist/)      |
                             |   - File type & size validation (< 25MB) |
                             |   - Instant local preview (ObjectURL)    |
                             |   - Scanning animation & state machine   |
                             +------------------------------------------+
                                                  |
                                   POST /api/detect (multipart/form-data)
                                                  v
                             +------------------------------------------+
                             |       FastAPI Backend (app/main.py)      |
                             |   - File format & byte verification      |
                             |   - OpenCV decoding check (cv2.imread)   |
                             |   - Singleton pipeline lifecycle         |
                             +------------------------------------------+
                                                  |
                                                  v
                             +------------------------------------------+
                             |  VehicleDamagePipeline (src/pipeline.py) |
                             |   - Coordinates detector inference       |
                             |   - Computes area coverage & metrics     |
                             |   - OpenCV colored annotation rendering  |
                             |   - Writes output to outputs/annotated_* |
                             +------------------------------------------+
                                                  |
                                                  v
                             +------------------------------------------+
                             |  VehicleDamageDetector (src/detector.py) |
                             |   - In-memory YOLOv8n (models/best.pt)   |
                             |   - Device auto-selection (CUDA / CPU)   |
                             |   - Thresholds: conf=0.25, iou=0.45      |
                             +------------------------------------------+
                                                  |
                                       JSON Response + Image URL
                                                  v
                             +------------------------------------------+
                             |             Results Dashboard            |
                             |   - Annotated image inspection viewer    |
                             |   - 1.3x Zoom & Fullscreen modal         |
                             |   - Severity metric cards & class badges |
                             |   - Formatted JSON diagnostic export     |
                             +------------------------------------------+
```

---

## Technology Stack

| Domain | Technology | Purpose |
| :--- | :--- | :--- |
| **Deep Learning Engine** | [Ultralytics YOLOv8](https://docs.ultralytics.com/) (`v8.3+`) | Object detection architecture, model loading, and inference |
| **Framework Runtime** | [PyTorch](https://pytorch.org/) (`v2.0+`) | Deep learning tensor computation on CPU and CUDA GPU |
| **Computer Vision** | [OpenCV](https://opencv.org/) (`opencv-python v4.8+`) | Image decode validation, bounding box rendering, and labeling |
| **Image Processing** | [Pillow](https://python-pillow.org/) & [NumPy](https://numpy.org/) | Array operations, coordinate math, and bounding area coverage |
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/) (`v0.100+`) | Asynchronous REST API, CORS middleware, and static hosting |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org/) (`v0.23+`) | Production-ready ASGI server running the FastAPI application |
| **Primary Frontend** | [React 19](https://react.dev/), [TypeScript](https://www.typescriptlang.org/), [Vite](https://vitejs.dev/) | Component-driven UI with type safety and optimized bundle |
| **Styling** | [Tailwind CSS](https://tailwindcss.com/) | Modern responsive design system with custom utility styling |
| **Fallback Frontend** | Vanilla HTML5 / JavaScript (ES6+) | Lightweight client requiring no build step (`app/frontend/`) |
| **Evaluation & Metrics** | [Matplotlib](https://matplotlib.org/), [Pandas](https://pandas.pydata.org/), [PyYAML](https://pyyaml.org/) | Benchmark computation, PR curve plotting, and report generation |
| **Testing Framework** | [Pytest](https://docs.pytest.org/) & [HTTPX](https://www.python-httpx.org/) | Automated unit and integration testing across 7 modules |

---

## Model Information & Benchmark Metrics

### Model Architecture
- **Base Architecture:** YOLOv8 Nano (`yolov8n`), single-stage anchor-free convolutional detector.
- **Backbone & Neck:** Modified CSPDarknet53 with C2f (Cross-Stage Partial with 2 convolutions) blocks, SPPF (Spatial Pyramid Pooling Fast), and PAN-FPN (Path Aggregation Feature Pyramid Network).
- **Detection Head:** Decoupled head separating classification and bounding box regression branches with Task-Aligned Assigner (TAL).
- **Parameters:** 3,012,018 (~3.01M parameters).
- **Checkpoint File:** `models/best.pt` (~5.95 MB, 6,234,154 bytes).
- **Checkpoint Checksum (SHA-256):** `59f7a958e23cd84777c0785626e2a4ef6d24911051702d8cc488ef89ec2ac75f`.
- **Training Setup:** Trained for 50 epochs with batch size 16 at $640 \times 640$ resolution using mixed precision (`amp=True`) on Google Colab.

### Benchmark Evaluation on Hold-Out Test Split
Evaluated across **374 test images** containing **785 ground truth damage instances** from the CarDD test set using `src/evaluation.py`:

#### Overall Performance
| Metric | Benchmark Score | Description |
| :--- | :---: | :--- |
| **Precision ($P$)** | **0.7520** (75.2%) | Ratio of correct damage predictions among all detections |
| **Recall ($R$)** | **0.6857** (68.6%) | Ratio of ground truth damage instances detected |
| **mAP@0.5** | **0.7214** (72.1%) | Mean Average Precision at IoU threshold of 0.50 |
| **mAP@0.5:0.95** | **0.5600** (56.0%) | Mean Average Precision averaged over IoU 0.50 to 0.95 |
| **Harmonic F1 Score** | **0.7173** (71.7%) | Harmonic mean of overall Precision and Recall ($2PR / (P+R)$) |
| **Macro Average F1** | **0.7147** (71.5%) | Arithmetic mean of optimal per-class F1 scores |

#### Per-Class Performance Breakdown
| Class ID | Class Name | Ground Truth Instances | Precision | Recall | Optimal F1 | mAP@0.5 | mAP@0.5:0.95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | `dent` | 236 | 0.6362 | 0.5854 | 0.6098 | 0.6312 | 0.3570 |
| `1` | `scratch` | 307 | 0.5877 | 0.5668 | 0.5771 | 0.5947 | 0.3503 |
| `2` | `crack` | 70 | 0.5749 | 0.4058 | 0.4757 | 0.4041 | 0.2044 |
| `3` | `glass shatter` | 71 | 0.9087 | 0.9816 | 0.9437 | 0.9881 | 0.9094 |
| `4` | `lamp broken` | 69 | 0.8399 | 0.7246 | 0.7780 | 0.8116 | 0.6708 |
| `5` | `tire flat` | 32 | 0.9645 | 0.8501 | 0.9037 | 0.8986 | 0.8684 |
| **All** | **Overall Summary** | **785** | **0.7520** | **0.6857** | **0.7173** | **0.7214** | **0.5600** |

*Key Findings:*
- `glass shatter` and `tire flat` achieve high precision and mAP@0.5 (>0.89) due to distinct structural patterns.
- `scratch` and `dent` account for >70% of dataset damage instances, yielding balanced F1 scores (~0.58–0.61).
- `crack` represents the most challenging category due to narrow geometries and visual similarity to deep scratches.

---

## Backend & REST API

The FastAPI server (`app/main.py`) provides asynchronous REST endpoints and serves the compiled frontend.

### Endpoints

| Method | Endpoint | Description | Response Type |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Serves the primary React 19 web application | `text/html` |
| `GET` | `/api/health` | Health check, active compute device (`cpu`/`cuda`), and classes | `application/json` |
| `POST` | `/api/detect` | Multipart file upload (`file`), runs inference and analytics | `application/json` |
| `GET` | `/api/results/{filename}` | Serves rendered annotated output images from `outputs/` | Image stream (`image/jpeg`, etc.) |
| `GET` | `/docs` | Interactive Swagger UI API documentation | `text/html` |
| `GET` | `/redoc` | Interactive ReDoc API documentation | `text/html` |

### Sample Detection Response (`POST /api/detect`)

```json
{
  "damage_status": "Damage Detected",
  "has_damage": true,
  "total_detections": 2,
  "unique_categories": ["dent", "tire flat"],
  "average_confidence": 0.842,
  "damage_area_percentage": 4.12,
  "image_width": 1024,
  "image_height": 768,
  "inference_time_ms": 48.6,
  "annotated_image": "/api/results/annotated_3a7b1c.jpg",
  "annotated_image_path": "outputs/annotated_3a7b1c.jpg",
  "detections": [
    {
      "class_id": 5,
      "class_name": "tire flat",
      "confidence": 0.895,
      "box": [140.2, 450.1, 380.6, 710.4],
      "box_normalized": [0.137, 0.586, 0.372, 0.925],
      "area_pixels": 62512.1,
      "area_percentage": 7.95
    },
    {
      "class_id": 0,
      "class_name": "dent",
      "confidence": 0.789,
      "box": [410.0, 310.5, 530.2, 420.0],
      "box_normalized": [0.400, 0.404, 0.518, 0.547],
      "area_pixels": 13161.4,
      "area_percentage": 1.67
    }
  ]
}
```

---

## Frontend Interfaces

The project provides two frontend implementations:

1. **React 19 Single-Page Application (Primary)**
   - Located in `Vehicle Damage Scanner UI/` and compiled into `app/frontend/dist/`.
   - Built with React 19, TypeScript, Vite, and Tailwind CSS.
   - Mounted as the default root `/` in `app/main.py`.
2. **Vanilla HTML5/JS Single-Page Application (Fallback)**
   - Located in `app/frontend/index.html`.
   - Styled with Tailwind CSS via CDN; requires zero npm build steps.
   - Can be served independently via `scripts/run_frontend.py` on port 3000.

---

## Repository Structure

```
vehicle-damage-scanner/
├── app/                               # Application layer
│   ├── frontend/                      # Web frontend assets
│   │   ├── dist/                      # Pre-bundled React 19 production build (served by FastAPI)
│   │   ├── samples/                   # Sample test images (000012.jpg, 000015.jpg)
│   │   └── index.html                 # Fallback standalone vanilla HTML5/JS frontend
│   ├── __init__.py
│   └── main.py                        # FastAPI REST server, endpoints, and static mount
├── dataset/                           # Local CarDD dataset (excluded from Git; ~2.83 GB)
│   ├── train/                         # Training images and YOLO annotations (2,816 samples)
│   ├── val/                           # Validation images and YOLO annotations (810 samples)
│   ├── test/                          # Test images and YOLO annotations (374 samples)
│   └── data.yaml                      # Dataset YAML configuration
├── evaluation/                        # Evaluation benchmarks and visual artifacts
│   ├── detection/
│   │   ├── plots/                     # PR curves, F1 curves, confusion matrices
│   │   ├── eval_dataset.yaml          # Local test split configuration
│   │   ├── report.md                  # Comprehensive benchmark report
│   │   ├── results.csv                # Tabular per-class metrics
│   │   └── results.json               # Structured evaluation metrics
│   └── .gitkeep
├── models/
│   └── best.pt                        # Trained YOLOv8n checkpoint weights (5.95 MB, included)
├── outputs/                           # Generated inference visualizations (annotated images)
│   └── .gitkeep
├── scripts/                           # Operational and administrative runners
│   ├── evaluate_model.py              # CLI test set evaluation runner
│   ├── run_api.py                     # Starts unified FastAPI server on port 8000
│   ├── run_detector_demo.py           # Quick CLI detector test on sample image
│   ├── run_frontend.py                # Standalone HTTP server for fallback frontend (port 3000)
│   └── run_pipeline_demo.py           # Quick CLI pipeline analysis demo
├── src/                               # Core Python ML and pipeline library
│   ├── __init__.py
│   ├── config.py                      # Application settings and environment variables
│   ├── detector.py                    # VehicleDamageDetector (YOLOv8 wrapper)
│   ├── evaluation.py                  # ModelEvaluator benchmarking engine
│   ├── model_loader.py                # Hardware detection and model loading utilities
│   └── pipeline.py                    # VehicleDamagePipeline (Analytics and OpenCV drawing)
├── tests/                             # Automated test suite (Pytest)
│   ├── fixtures/                      # Test sample images (000012.jpg)
│   ├── test_api.py                    # FastAPI endpoint tests and upload validation
│   ├── test_config_and_pruning.py     # Configuration parsing and output retention tests
│   ├── test_detector.py               # Detector loading and inference unit tests
│   ├── test_evaluation.py             # Evaluation structures and metric calculation tests
│   ├── test_frontend_static.py        # Frontend markup and accessibility tests
│   ├── test_model_loading.py          # Weight file integrity and CUDA detection tests
│   └── test_pipeline.py               # End-to-end pipeline and CV2 annotation tests
├── Vehicle Damage Scanner UI/         # React 19 + TypeScript + Vite source code
│   ├── src/                           # React components and styling (App.tsx, index.css)
│   ├── package.json                   # Frontend dependencies
│   ├── tsconfig.json                  # TypeScript configuration
│   └── vite.config.ts                 # Vite build configuration (outputs to app/frontend/dist)
├── .env.example                       # Environment configuration template
├── .gitignore                         # Git exclusion rules
├── PROJECT_HANDOFF.md                 # In-depth architectural audit and handoff reference
├── README.md                          # Project documentation
└── requirements.txt                   # Python dependencies
```

---

## Installation & Setup

### Prerequisites
- **Python:** Version 3.10 or higher (tested on Python 3.11).
- **Node.js & npm:** Version 18 or higher (*optional*, only needed if you modify and recompile the React UI).
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/Rudra4604/vehicle-damage-scanner.git
cd vehicle-damage-scanner
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Optional)
The project works out of the box with default settings. To customize configurations:
```bash
cp .env.example .env
```
Key configurable variables in `.env`:
- `HOST`: Server host (default: `127.0.0.1`).
- `PORT`: Server port (default: `8000`).
- `MODEL_PATH`: Path to model weights (default: `models/best.pt`).
- `CONFIDENCE_THRESHOLD`: Minimum detection confidence (default: `0.25`).
- `IOU_THRESHOLD`: Non-Maximum Suppression IoU threshold (default: `0.45`).
- `OUTPUT_RETENTION_HOURS`: Automatic pruning age for generated output images (default: `24`).

---

## Running the Application

### 1. Start the Unified Server (Recommended)
Launch the FastAPI backend, which also serves the compiled React frontend:
```bash
python scripts/run_api.py
```
*Alternatively, run with Uvicorn directly:*
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Once started, open your browser:
- **Web Application:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive API Docs (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **API Health Check:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### 2. Run CLI Demos (Optional)
Test model loading and inference from the command line:

```bash
# Verify model weights, hardware device, and run sample inference
python src/model_loader.py

# Run standalone detector demo on a sample image
python scripts/run_detector_demo.py

# Run pipeline demo with damage analytics
python scripts/run_pipeline_demo.py
```

---

## Building & Developing the Frontend

The compiled React production bundle is already included in `app/frontend/dist/`, so you **do not** need Node.js to run the web application.

If you wish to modify or extend the React UI:

```bash
cd "Vehicle Damage Scanner UI"

# Install frontend dependencies
npm install

# Start the Vite development server with Hot Module Replacement (HMR)
npm run dev

# Compile the production bundle (automatically builds into ../app/frontend/dist)
npm run build
```

---

## How to Use the Application

1. **Open the Web Interface:** Navigate to `http://127.0.0.1:8000/` in your browser.
2. **Upload a Vehicle Image:**
   - Drag and drop an exterior car image (`.jpg`, `.png`, `.webp`, `.bmp`) into the upload dropzone.
   - Or click **Browse Files** to select a photo.
   - Sample images are available in `app/frontend/samples/` (e.g., `000012.jpg` showing a dent and flat tire).
3. **Execute Analysis:** Click **Analyze Damage**. The UI will display a scanning sweep animation while the backend processes the image.
4. **Inspect Results:**
   - **Visual Overlay:** View the annotated vehicle photo with color-coded bounding boxes and confidence labels.
   - **Interactive Controls:** Use the **1.3x Zoom** toggle or click **Fullscreen** for detailed inspection.
   - **Severity Metrics:** Review the Total Detections, Damaged Area Coverage percentage, and Average Confidence.
   - **Damage Breakdown:** Examine individual damage cards detailing the category and bounding box coordinates.
5. **Export Report:** Click **Export JSON** to download a structured JSON diagnostic summary for insurance or maintenance records.

---

## Testing Suite

The repository includes a comprehensive automated test suite built with **Pytest** and **HTTPX**:

```bash
# Run all 39 unit and integration tests
pytest

# Run tests with verbose output
pytest -v
```

### Test Coverage Breakdown
- `tests/test_api.py`: FastAPI health endpoints, valid image uploads, unsupported file formats, empty files, corrupt image decoding, and annotated image retrieval.
- `tests/test_detector.py`: YOLOv8 model loading, prediction structures, threshold filtering, and class mapping.
- `tests/test_pipeline.py`: End-to-end pipeline execution, bounding box scaling, area calculation math, OpenCV drawing, and dictionary serialization.
- `tests/test_model_loading.py`: Weight file existence, parameter count assertions, SHA-256 integrity, and compute device selection.
- `tests/test_evaluation.py`: ModelEvaluator configuration generation and dataclass serialization.
- `tests/test_config_and_pruning.py`: Environment variable loading and output directory retention pruning.
- `tests/test_frontend_static.py`: HTML markup structure, element IDs, and accessibility compliance.

---

## Important Notes on Dataset & Model Weights

- **Model Weights Included:** The trained model weights file (`models/best.pt`, 5.95 MB) is tracked in the repository. Running the web application, REST API, CLI demos, and automated test suite works immediately without downloading external checkpoints.
- **Dataset Exclusion:** The raw CarDD dataset (`dataset/`, ~2.83 GB, 4,000 images) is excluded from Git via `.gitignore` to keep the repository lightweight and responsive.
- **Running Full Dataset Re-evaluation:** If you have the CarDD dataset placed in `dataset/`, you can re-run the complete test set evaluation and regenerate benchmark plots using:
  ```bash
  python scripts/evaluate_model.py
  ```

---

## License & Attribution

This project utilizes the **CarDD (Car Damage Dataset)** and the **Ultralytics YOLOv8** architecture. Developed as a deep learning and computer vision engineering project.
