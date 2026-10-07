# PROJECT HANDOFF DOCUMENT: VEHICLE DAMAGE DETECTION SYSTEM

> **Document Version:** 1.0.0  
> **Date of Audit:** October 6, 2026  
> **Target Audience:** Future AI Assistant (ChatGPT) / Software Engineer Handoff  
> **Source Directory:** `c:\Users\rudra\Desktop\DLL Project`  
> **Test Status:** 31 / 31 passing unit & integration tests (`pytest tests/ -v`)

---

## 1. PROJECT OVERVIEW

### Explain This Project to Another AI
This project is an end-to-end computer vision and deep learning application for automated vehicle exterior damage detection and classification. It uses an in-memory YOLOv8 Nano (`yolov8n`) model trained on the CarDD (Car Damage Dataset) to localize and classify 6 distinct categories of vehicular damage: **dent**, **scratch**, **crack**, **glass shatter**, **lamp broken**, and **tire flat**. 

The system consists of:
1. A trained PyTorch/Ultralytics weights checkpoint (`models/best.pt`, 3.01M parameters).
2. A core inference and evaluation library (`src/model_loader.py`, `src/detector.py`, `src/pipeline.py`, `src/evaluation.py`).
3. A FastAPI asynchronous backend (`app/main.py`, served via Uvicorn on port 8000) providing REST endpoints for system health, multipart file upload damage detection, and annotated image retrieval.
4. A static single-page frontend (`app/frontend/index.html`, served on port 3000) crafted with Tailwind CSS and vanilla JavaScript featuring drag-and-drop image uploads, live scanning animations, bounding-box overlay inspection with zoom/fullscreen modal, confidence metric breakdowns, and JSON export.
5. A comprehensive evaluation suite producing mAP/F1 metrics, confusion matrices, and precision-recall curves matching Colab benchmarks.

### Key Metadata
- **Project Name:** DLL Project — Vehicle Damage Scanner
- **Problem Solved:** Automates manual, error-prone vehicle damage appraisal for insurance claim pre-assessment, rental car return check-ins, and fleet management.
- **Target Users:** Insurance adjusters, car rental inspection staff, collision repair estimators, and vehicle owners filing claims.
- **Main Functionality:** Accepts an exterior car photograph, detects and classifies damaged regions with bounding boxes, calculates percentage of damaged area relative to the image, computes confidence metrics, draws color-coded bounding boxes with labels, and outputs both an annotated image and a structured JSON payload.
- **Current Project Status:** Highly functional core pipeline, verified evaluation metrics, working FastAPI backend, complete interactive UI, and 31 automated tests. Missing containerization (Docker), deployment configurations, and production persistence/database layers.
- **What Makes the Project Unique:** End-to-end independence—the model loads as a singleton in memory; inference pipeline computes damage area coverage in pixels and percentage; bounding boxes are rendered with custom per-class color palettes; full reproducibility audit logs are checked against Kaggle/Colab training runs.
- **Final Intended Product:** A production-grade vehicle damage assessment microservice and web application capable of running locally or in cloud containers with GPU acceleration.

---

## 2. COMPLETE TECHNOLOGY STACK

All technologies listed below have been verified directly from code, lock files, and runtime tests:

| Category | Technology | Version / Spec | Usage Location & Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.11.1 (Windows x64) | Entire backend, inference engine, evaluation scripts, and test suite. |
| **Language** | JavaScript (ES6+) | Vanilla (No Node build) | Client-side reactive logic in `app/frontend/index.html`. |
| **Language** | HTML5 / CSS3 | Standard HTML5 | UI layout and custom CSS keyframes/shimmer animations in `app/frontend/index.html`. |
| **Deep Learning Framework** | PyTorch (`torch`) | $\ge$ 2.0.0 (active 2.6.0) | Neural network runtime and tensor execution engine. |
| **Computer Vision Engine** | Ultralytics (`ultralytics`) | $\ge$ 8.0.0 (active 8.3.28) | YOLOv8 model architecture, weights loading, inference engine, and evaluation validator. |
| **Vision Utilities** | OpenCV (`opencv-python`) | $\ge$ 4.8.0 | Image decoding validation, drawing colored bounding boxes, text banner rendering (`src/pipeline.py`, `app/main.py`). |
| **Image Library** | Pillow (`PIL`) | $\ge$ 10.0.0 | Image file validation and test assertions. |
| **Numerical Processing** | NumPy (`numpy`) | $\ge$ 1.24.0 | Coordinate calculations, bounding box area math, confidence mean averaging. |
| **Data Processing** | Pandas (`pandas`) | Integrated via Ultralytics | Generating tabular benchmark data and saving `evaluation/detection/results.csv`. |
| **Plotting & Visualization** | Matplotlib (`matplotlib`) | $\ge$ 3.7.0 | Plotting per-class performance bar charts (`src/evaluation.py`). |
| **Configuration Parsing** | PyYAML (`pyyaml`) | $\ge$ 6.0 | Parsing and generating dataset configs (`data.yaml`, `eval_dataset.yaml`). |
| **Backend API Framework** | FastAPI (`fastapi`) | $\ge$ 0.100.0 | High-performance asynchronous REST API framework (`app/main.py`). |
| **ASGI Web Server** | Uvicorn (`uvicorn`) | $\ge$ 0.23.0 | Running FastAPI server with auto-reload (`scripts/run_api.py`). |
| **Form/Upload Support** | `python-multipart` | $\ge$ 0.0.6 | Streaming file upload parsing in FastAPI (`UploadFile`). |
| **HTTP Client (Testing)** | HTTPX (`httpx`) | $\ge$ 0.24.0 | Driving `TestClient` in `tests/test_api.py`. |
| **Testing Framework** | Pytest (`pytest`) | $\ge$ 7.0.0 (active 9.1.1) | Automated test execution across 6 test modules. |
| **Frontend Styling** | Tailwind CSS CDN | Play CDN (`cdn.tailwindcss.com`) | Design system styling using custom color palette tokens in `index.html`. |
| **Typography & Icons** | Google Fonts & Material Symbols | Web fonts | `Plus Jakarta Sans` typography and Google `Material Symbols Outlined` icons. |
| **Local Web Server** | Python `http.server` | Built-in | Static web server running on port 3000 via `scripts/run_frontend.py`. |

---

## 3. COMPLETE FOLDER STRUCTURE

```
c:\Users\rudra\Desktop\DLL Project\
├── .pytest_cache/                     # Pytest execution cache
│   ├── v/
│   ├── .gitignore
│   ├── CACHEDIR.TAG
│   └── README.md
├── app/                               # Web application backend and frontend
│   ├── frontend/                      # User-facing client assets
│   │   ├── samples/                   # Sample images for testing & UI demonstrations
│   │   │   ├── 000012.jpg             # Test image with flat tire & dent (1,016 KB)
│   │   │   ├── 000015.jpg             # Test image sample (430 KB)
│   │   │   └── document.txt           # Non-image test fixture for error handling
│   │   ├── stitch_vehicle_damage_scanner/     # Stitch design artifact iteration 0
│   │   │   ├── code.html              # Reference prototype HTML
│   │   │   ├── DESIGN.md              # Design tokens and style guide specifications
│   │   │   └── screen.png             # UI mockup visual render
│   │   ├── stitch_vehicle_damage_scanner (1)/ # Stitch design artifact iteration 1
│   │   │   ├── code.html
│   │   │   ├── DESIGN.md
│   │   │   └── screen.png
│   │   ├── stitch_vehicle_damage_scanner (2)/ # Stitch design artifact iteration 2
│   │   │   ├── code.html
│   │   │   ├── DESIGN.md
│   │   │   └── screen.png
│   │   └── index.html                 # Production-ready single page web application (63 KB)
│   ├── __pycache__/                   # Compiled Python bytecode
│   ├── .gitkeep                       # Directory retention placeholder
│   ├── __init__.py                    # App package initialization
│   └── main.py                        # FastAPI application REST server & endpoint router
├── DataSEt/                           # CarDD Car Damage Dataset (Converted to YOLO format)
│   ├── models/                        # Checkpoint storage within dataset directory
│   │   └── best.pt                    # Duplicate model weights (SHA-256 match, 5.95 MB)
│   ├── test/                          # Hold-out test split (374 images, 374 label files)
│   │   ├── images/                    # 374 .jpg car images
│   │   └── labels/                    # 374 .txt YOLO bounding box label files
│   ├── train/                         # Training split (2,816 images, 2,816 label files)
│   │   ├── images/                    # 2,816 .jpg car images
│   │   └── labels/                    # 2,816 .txt YOLO bounding box label files
│   ├── val/                           # Validation split (810 images, 810 label files)
│   │   ├── images/                    # 810 .jpg car images
│   │   └── labels/                    # 810 .txt YOLO bounding box label files
│   └── data.yaml                      # Original Kaggle training dataset configuration file
├── evaluation/                        # Evaluation outputs and benchmarks
│   ├── detection/                     # Object detection evaluation outputs
│   │   ├── plots/                     # Visual performance charts
│   │   │   ├── confusion_matrix.png            # Raw count confusion matrix
│   │   │   ├── confusion_matrix_normalized.png # Normalized confusion matrix
│   │   │   ├── per_class_metrics.png          # Grouped bar chart (P, R, F1, mAP50)
│   │   │   ├── val_batch0_labels.jpg          # Ground truth validation grid
│   │   │   └── val_batch0_pred.jpg            # Predicted validation grid
│   │   ├── eval_dataset.yaml          # Localized POSIX path dataset YAML for evaluation
│   │   ├── report.md                  # Detailed Markdown evaluation report
│   │   ├── results.csv                # Tabular per-class and overall metrics CSV
│   │   └── results.json               # Structured JSON evaluation metrics
│   └── .gitkeep                       # Evaluation directory retention placeholder
├── models/                            # Model weight store (Primary)
│   ├── best.pt                        # Trained YOLOv8n checkpoint (SHA-256: 59f7a95..., 5.95 MB)
│   └── best.pt.backup                 # Integrity backup of best.pt (Byte-identical)
├── outputs/                           # Generated inference visualizations (Persisted)
│   ├── .gitkeep                       # Directory retention placeholder
│   └── annotated_*.jpg / *.png        # Rendered output images with bounding boxes & tags
├── scripts/                           # Operational and administrative CLI scripts
│   ├── evaluate_model.py              # Runs ModelEvaluator on the test dataset split
│   ├── run_api.py                     # Starts Uvicorn server for FastAPI backend (port 8000)
│   ├── run_frontend.py                # Starts ThreadingHTTPServer for frontend (port 3000)
│   ├── test_detector.py               # Standalone verification runner for VehicleDamageDetector
│   └── test_pipeline.py               # Standalone verification runner for VehicleDamagePipeline
├── src/                               # Core application & machine learning logic
│   ├── __pycache__/                   # Compiled Python bytecode
│   ├── __init__.py                    # src package root
│   ├── detector.py                    # VehicleDamageDetector (YOLOv8 wrapper, in-memory)
│   ├── evaluation.py                  # ModelEvaluator & evaluation data structures
│   ├── model_loader.py                # Hardware detection, model locator, and sample inference
│   └── pipeline.py                    # VehicleDamagePipeline (Analytics, drawing, bounding boxes)
├── tests/                             # Automated test suite (Pytest)
│   ├── __pycache__/                   # Compiled test bytecode
│   ├── test_api.py                    # 8 tests: FastAPI endpoints, uploads, error handling
│   ├── test_detector.py               # 7 tests: VehicleDamageDetector loading & predictions
│   ├── test_evaluation.py             # 3 tests: ModelEvaluator config & dataclass serialization
│   ├── test_frontend_static.py        # 1 test: HTML structure, IDs, and UI accessibility
│   ├── test_model_loading.py          # 4 tests: Weights integrity, class names, CUDA detection
│   └── test_pipeline.py               # 8 tests: End-to-end pipeline analysis & CV2 annotations
├── README.md                          # Initial project documentation
└── requirements.txt                   # Production and testing Python dependencies
```

### File Usage & Dependencies Matrix
- `src/detector.py` is imported by `src/pipeline.py`, `scripts/test_detector.py`, and `tests/test_detector.py`.
- `src/pipeline.py` is imported by `app/main.py`, `scripts/test_pipeline.py`, and `tests/test_pipeline.py`.
- `src/model_loader.py` is imported by `tests/test_model_loading.py` and run standalone.
- `src/evaluation.py` is imported by `scripts/evaluate_model.py` and `tests/test_evaluation.py`.
- `app/main.py` is run by `scripts/run_api.py` and tested by `tests/test_api.py`.
- `app/frontend/index.html` communicates with `app/main.py` via HTTP fetch calls to `http://127.0.0.1:8000/api/...`.

---

## 4. APPLICATION ARCHITECTURE

### System Flow Diagram
```
                       +---------------------------------------+
                       |              USER BROWSER             |
                       +---------------------------------------+
                                          |
                      Upload Image / Drag & Drop / Paste
                                          v
      +-----------------------------------------------------------------------+
      | FRONTEND CLIENT (app/frontend/index.html on http://127.0.0.1:3000)    |
      | - Pre-upload extension & file-size validation (max 25MB)             |
      | - ObjectURL local preview rendering                                   |
      | - Multi-step state machine: [Inspect] -> [Analyzing] -> [Results]     |
      | - Progress shimmer & animated scan line sweep                         |
      +-----------------------------------------------------------------------+
                                          |
                        HTTP POST /api/detect (multipart/form-data)
                                          v
      +-----------------------------------------------------------------------+
      | FASTAPI BACKEND (app/main.py on http://127.0.0.1:8000 via Uvicorn)    |
      | - CORS Middleware (allows origins http://localhost:3000, etc.)       |
      | - Lifespan Manager: loads singleton VehicleDamagePipeline on startup  |
      | - Stream validation: checks empty payloads & allowed image formats    |
      | - OpenCV decoding verification (cv2.imread)                           |
      +-----------------------------------------------------------------------+
                                          |
                               Calls analyze_image()
                                          v
      +-----------------------------------------------------------------------+
      | VEHICLE DAMAGE PIPELINE (src/pipeline.py)                             |
      | - Delegates to VehicleDamageDetector.predict()                        |
      | - Computes high-level aggregated damage metrics:                      |
      |     * total_detections, unique_categories                             |
      |     * average_confidence, highest_confidence_detection                |
      |     * total_damaged_area (px²), damage_area_percentage (% of image)   |
      +-----------------------------------------------------------------------+
                                          |
                               Calls model.predict()
                                          v
      +-----------------------------------------------------------------------+
      | VEHICLE DAMAGE DETECTOR (src/detector.py)                             |
      | - Ultralytics YOLOv8n in-memory instance                              |
      | - Hardware Device: Auto-selects CUDA GPU (if available) else CPU      |
      | - Runs forward inference with conf=0.25, iou=0.45                     |
      | - Extracts normalized boxes, class IDs, class names, and scores       |
      +-----------------------------------------------------------------------+
                                          |
                               Weights loaded from disk
                                          v
                             [ models/best.pt ] (5.95 MB)
                                          |
                                 Returns raw boxes
                                          v
      +-----------------------------------------------------------------------+
      | OPENCV ANNOTATION ENGINE (src/pipeline.py)                            |
      | - Renders color-coded 2px bounding boxes (per-class BGR palette)      |
      | - Computes text size and draws filled label background rectangle      |
      | - Writes anti-aliased class label and confidence percentage           |
      | - Writes rendered image to outputs/annotated_<uuid>.<ext>             |
      +-----------------------------------------------------------------------+
                                          |
                              JSON Response + Image Path
                                          v
      +-----------------------------------------------------------------------+
      | API RESPONSE (JSON)                                                   |
      | {                                                                     |
      |   "damage_status": "Damage Detected",                                 |
      |   "has_damage": true,                                                 |
      |   "total_detections": 2,                                              |
      |   "unique_categories": ["dent", "tire flat"],                         |
      |   "average_confidence": 0.842,                                        |
      |   "damage_area_percentage": 4.12,                                     |
      |   "detections": [...],                                                |
      |   "annotated_image": "/api/results/annotated_1f025746aec8.jpg"        |
      | }                                                                     |
      +-----------------------------------------------------------------------+
                                          |
                    Frontend fetches /api/results/annotated_...
                                          v
      +-----------------------------------------------------------------------+
      | RESULTS VIEW (app/frontend/index.html)                                |
      | - Displays high-resolution annotated image in inspection viewer       |
      | - Interactive controls: 1.3x Zoom toggle & Fullscreen Modal           |
      | - Metric cards: Total Detections, Damaged Area %, Avg Confidence      |
      | - Identified damage cards with class badges and bounding box coords   |
      | - Export report as formatted JSON file                                |
      +-----------------------------------------------------------------------+
```

---

## 5. DEEP LEARNING / MACHINE LEARNING AUDIT

### 5.1 Model Specifications
- **Model Architecture:** YOLOv8n (YOLOv8 Nano Detection Model).
- **Architecture Family:** Single-stage anchor-free convolutional object detector with decoupled head (independent loss branches for bounding box regression and classification).
- **Backbone & Neck:** Modified CSPDarknet53 with C2f (Cross-Stage Partial with 2 convolutions) modules and SPPF (Spatial Pyramid Pooling Fast). PAN-FPN (Path Aggregation Feature Pyramid Network) neck.
- **Head:** Anchor-free decoupled detection head using Task-Aligned Assigner (TAL).
- **Total Parameters:** **3,012,018** (~3.01 million parameters).
- **Trainable Parameters:** 0 in inference mode (frozen weights).
- **Model Checkpoint Size:** 6,234,154 bytes (~5.95 MB).
- **Primary Checkpoint Path:** `models/best.pt`
- **Backup Checkpoint Paths:** `models/best.pt.backup`, `DataSEt/models/best.pt`.
- **SHA-256 Checksum:** `59f7a958e23cd84777c0785626e2a4ef6d24911051702d8cc488ef89ec2ac75f` (All 3 copies are byte-identical).
- **Base Pretrained Weights:** `yolov8n.pt` (transfer learning initialized from COCO pretrained weights).
- **Input Shape:** Default $640 \times 640 \times 3$ (RGB normalized dynamically by Ultralytics during preprocessing).
- **Output Shape:** Tensor of shape `[1, 10, 8400]` (where 10 = 4 box coordinates $[x, y, w, h]$ + 6 class probabilities, evaluated across 8,400 multi-scale anchor points).
- **Activation Functions:** SiLU (Sigmoid Linear Unit / Swish-1) throughout backbone and neck. Softmax/Sigmoid across class outputs.
- **Loss Functions (Training):**
  - Bounding Box Regression: Complete IoU (CIoU) Loss combined with Distribution Focal Loss (DFL) (`box=7.5`, `dfl=1.5`).
  - Classification: Binary Cross Entropy (BCE) Loss (`cls=0.5`).

### 5.2 Training Hyperparameters (Extracted from Checkpoint)
The model was trained in Google Colab using Ultralytics. The following hyperparameters are recorded in `models/best.pt`:
- **Training Engine:** Ultralytics YOLOv8 CLI / Python API.
- **Base Model:** `yolov8n.pt`
- **Dataset Path at Train Time:** `/content/CarDD/data_local.yaml`
- **Output Project:** `/content/drive/MyDrive/VehicleDamage/CarDD/training_results_v2`
- **Run Name:** `car_damage_yolov8n`
- **Epochs:** 50
- **Batch Size:** 16
- **Image Size (`imgsz`):** 640
- **Optimizer:** `auto` (resolved to SGD with momentum or AdamW based on dataset scale)
- **Initial Learning Rate (`lr0`):** 0.01
- **Final Learning Rate Factor (`lrf`):** 0.01
- **Momentum:** 0.937
- **Weight Decay:** 0.0005
- **Warmup Epochs:** 3.0
- **Warmup Momentum:** 0.8
- **Augmentation Pipeline:**
  - Mosaic: 1.0 (disabled during final 10 epochs via `close_mosaic=10`)
  - Horizontal Flip (`fliplr`): 0.5
  - Vertical Flip (`flipud`): 0.0
  - Random Erasing: 0.4
  - Color Jitter (HSV): Hue fraction 0.015, Saturation 0.7, Value 0.4
  - Translation: 0.1, Scaling: 0.5
  - Auto-Augment: `randaugment`
  - Mixed Precision (`amp`): True
  - Workers: 8

### 5.3 Classes & Dataset Characteristics
The dataset is the **CarDD (Car Damage Dataset)** converted to YOLO standard format.
- **Dataset Path in Project:** `DataSEt/`
- **Total Images:** 4,000 images across 3 splits.
- **Total Annotated Damage Instances:** 8,740 bounding boxes.

#### Split Breakdown
| Split | Image Count | Label Files | Total Damage Instances |
| :--- | :---: | :---: | :---: |
| **Train** | 2,816 | 2,816 | 6,211 |
| **Validation** | 810 | 810 | 1,744 |
| **Test** | 374 | 374 | 785 |
| **Total** | **4,000** | **4,000** | **8,740** |

#### Class Distribution Across Splits
| Class ID | Class Name | Train Instances | Val Instances | Test Instances | Total Dataset Instances | Class Share |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `0` | `dent` | 1,806 | 501 | 236 | 2,543 | 29.1% |
| `1` | `scratch` | 2,560 | 728 | 307 | 3,595 | 41.1% |
| `2` | `crack` | 651 | 177 | 70 | 898 | 10.3% |
| `3` | `glass shatter` | 475 | 135 | 71 | 681 | 7.8% |
| `4` | `lamp broken` | 494 | 141 | 69 | 704 | 8.1% |
| `5` | `tire flat` | 225 | 62 | 32 | 319 | 3.6% |
| **All** | **All Classes** | **6,211** | **1,744** | **785** | **8,740** | **100.0%** |

*Note on Class Imbalance:* `scratch` (41.1%) and `dent` (29.1%) represent over 70% of all damage samples. Rare classes like `tire flat` (3.6%) have lower representation, yet high precision/recall due to distinct geometric characteristics.

### 5.4 Test Set Benchmark & Reproducibility Audit
Independent evaluation was conducted across all 374 test set images (785 ground truth bounding boxes) using `src/evaluation.py`. The results reproduce the Colab benchmark:

| Metric | Colab Benchmark | Independent Evaluation | Discrepancy ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Precision ($P$)** | 0.7530 | 0.7520 | -0.0010 | Verified Match |
| **Recall ($R$)** | 0.6860 | 0.6857 | -0.0003 | Verified Match |
| **mAP@0.5** | 0.7219 | 0.7214 | -0.0005 | Verified Match |
| **mAP@0.5:0.95** | 0.5602 | 0.5600 | -0.0002 | Verified Match |
| **Harmonic F1 Score** | — | **0.7173** | — | Calculated ($2PR / (P+R)$) |
| **Macro Average F1** | — | **0.7147** | — | Arithmetic mean across classes |

#### Detailed Per-Class Performance on Test Set
| Class ID | Class Name | Instances | Precision | Recall | Optimal F1 | mAP@0.5 | mAP@0.5:0.95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | `dent` | 236 | 0.6362 | 0.5854 | 0.6098 | 0.6312 | 0.3570 |
| `1` | `scratch` | 307 | 0.5877 | 0.5668 | 0.5771 | 0.5947 | 0.3503 |
| `2` | `crack` | 70 | 0.5749 | 0.4058 | 0.4757 | 0.4041 | 0.2044 |
| `3` | `glass shatter` | 71 | 0.9087 | 0.9816 | 0.9437 | 0.9881 | 0.9094 |
| `4` | `lamp broken` | 69 | 0.8399 | 0.7246 | 0.7780 | 0.8116 | 0.6708 |
| `5` | `tire flat` | 32 | 0.9645 | 0.8501 | 0.9037 | 0.8986 | 0.8684 |
| **ALL** | **Overall** | **785** | **0.7520** | **0.6857** | **0.7173** | **0.7214** | **0.5600** |

*Analysis:*
- `glass shatter` achieves the highest mAP@0.5 (0.9881) and F1 (0.9437) due to distinct spider-web texture patterns.
- `tire flat` achieves 0.9645 Precision and 0.8986 mAP@0.5.
- `crack` is the hardest class (mAP@0.5 of 0.4041, Recall 0.4058) because cracks are thin, easily confused with deep scratches, and vary widely across bumper and windshield contours.

### 5.5 Inference Pipeline Flow
1. **Input Received:** File path or uploaded bytes.
2. **Pre-processing:** Image verified via OpenCV, dimensions $(H, W)$ cached. Model inference automatically executes letterbox resizing to $640 \times 640$, tensor conversion, and normalization to $[0.0, 1.0]$.
3. **Execution:** Loaded singleton YOLO model executes `predict(source, conf=0.25, iou=0.45, device=device)`.
4. **Post-processing:**
   - Boxes scaled back from $640 \times 640$ to original $(W, H)$ coordinates $[x_1, y_1, x_2, y_2]$.
   - Bounding box area $(x_2 - x_1) \times (y_2 - y_1)$ and normalized area relative to image calculated.
   - Total damaged area and damage area percentage calculated.
5. **Annotation Generation:** `cv2.rectangle` draws colored bounding box; text label background drawn; white text string rendered (`{class_name} {confidence:.1%}`).
6. **Artifact Persistence:** Annotated image written to `outputs/annotated_<uuid>.<ext>`.

---

## 6. COMPLETE DATA FLOW TRACING

### User Upload & Prediction Workflow
```
Step 1: User selects/drops file in browser (e.g., 000012.jpg, 1.01 MB).
        ├── JS checks extension (.jpg, .png, .webp, .bmp, .tiff)
        ├── JS checks file.size <= 25MB
        └── Local preview displayed instantly using URL.createObjectURL(file)

Step 2: User clicks "Analyze Damage" button.
        ├── UI transitions to Screen 2 [Analyzing]
        ├── Shimmer progress bar and scan sweep animation start
        ├── FormData object constructed with 'file' field
        └── Asynchronous fetch('POST http://127.0.0.1:8000/api/detect') dispatched

Step 3: FastAPI Backend receives multipart request.
        ├── Validates filename and extension against SUPPORTED_EXTENSIONS
        ├── Reads byte content and rejects empty files (0 bytes)
        ├── Writes bytes to temporary file in OS tempdir: tempfile.gettempdir()/upload_<uuid>.ext
        ├── Opens temp file using cv2.imread(temp_path)
        │   └── If decoding fails (corrupted data), raises HTTP 400 Bad Request
        └── Passes verified temp_path to VehicleDamagePipeline.analyze_image()

Step 4: VehicleDamagePipeline & VehicleDamageDetector execute.
        ├── Predicts boxes using in-memory model (conf=0.25, iou=0.45)
        ├── Calculates total_damaged_area and damage_area_percentage
        ├── Generates annotated image using OpenCV with class-specific colors
        ├── Writes annotated file to outputs/annotated_<uuid>.ext
        └── Unlinks the temporary upload file in a `finally:` block

Step 5: Backend constructs JSON response.
        ├── Sets annotated_image URL to "/api/results/annotated_<uuid>.ext"
        └── Returns HTTP 200 with structured JSON body

Step 6: Frontend processes response.
        ├── Sets progress bar to 100%
        ├── Transitions view to Screen 3 [Results]
        ├── Loads annotated image via GET http://127.0.0.1:8000/api/results/...
        ├── Populates metric cards (Total Detections, Damaged Area %, Avg Confidence)
        └── Renders interactive cards for each detection with coordinates and confidence bars
```

---

## 7. FRONTEND ARCHITECTURE & IMPLEMENTATION

### Files Controlling the UI
- **Primary Application File:** `app/frontend/index.html` (63,212 bytes, 1,225 lines).
- **Design Specifications:** `app/frontend/stitch_vehicle_damage_scanner/DESIGN.md`.
- **Sample Assets:** `app/frontend/samples/` (`000012.jpg`, `000015.jpg`, `document.txt`).
- **Static Runner:** `scripts/run_frontend.py` (serves `app/frontend/` on `http://127.0.0.1:3000`).

### UI Structure & Screens
The frontend is a reactive Single-Page Application (SPA) driven by vanilla JavaScript without complex build tools or npm dependencies:

1. **Persistent Header:**
   - Automotive logo mark (`directions_car` icon).
   - Navigation pill showing step progress: `[Inspect] -> [Analyzing] -> [Results]`.
   - Backend health heartbeat indicator dot (polls `GET /api/health`).

2. **Screen 1: Upload / Inspect View (`id="view-upload"`)**
   - Centered large dropzone (`id="dropzone"`) with dashed border.
   - Supports: Click-to-browse (`<input type="file">`), drag-and-drop, and direct clipboard image paste (`Ctrl+V`).
   - Image preview overlay (`id="dropzone-preview-overlay"`) displaying filename, size in MB, and resolution in pixels.
   - Action CTA button (`id="btn-analyze"`): Disabled grey by default; activates to green (`#316342`) when a valid image is staged.
   - Photography guidance cards (Good lighting, Wide angle, Clean surface).
   - Global error banner (`id="upload-error-banner"`).

3. **Screen 2: Analyzing / Inspection State View (`id="view-analyzing"`)**
   - Active preview canvas showing uploaded vehicle photo.
   - Smooth animated green scanning line sweep (`id="scanning-sweep"`).
   - Shimmering progress bar simulating diagnostic stages (Surface scan, Rim inspection, Bounding box assembly).
   - "Cancel & choose a different photo" button (`id="btn-cancel"`).

4. **Screen 3: Damage Assessment / Results View (`id="view-results"`)**
   - High-resolution annotated image viewer (`id="results-annotated-img"`).
   - Zoom toggle button (`1.3x` scale toggle via CSS transform).
   - Fullscreen modal overlay (`id="fullscreen-modal"`).
   - Overall condition badge ("Damage Detected" vs "Clean Vehicle").
   - 3 prominent metric cards: Total Detections, Damaged Area %, Average Confidence %.
   - Identified damage items list: Horizontal cards displaying class badge, icon, bounding box coordinates `[x1, y1, x2, y2]`, pixel area, and confidence progress bar.
   - Fallback zero-damage card (`id="results-zero-damage-card"`).
   - Actions: "Analyze Another Photo" (resets state) and "Download JSON" (exports payload as file).

---

## 8. BACKEND ARCHITECTURE & REST API

### Server Configuration
- **Entry Point:** `app/main.py`
- **Application Runner:** `scripts/run_api.py` (invokes `uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)`)
- **Framework:** FastAPI `1.0.0`
- **Lifespan Manager:** `@asynccontextmanager` initializes the singleton `VehicleDamagePipeline` during server startup, loading the model into GPU/CPU memory once.
- **CORS Middleware:** Allows cross-origin requests from `http://localhost:3000`, `http://127.0.0.1:3000`, `http://localhost:5173`, `http://127.0.0.1:5173`.

### API Endpoints Specification

| Method | Endpoint | Purpose | Request Input | Response Output | Status Codes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service liveness & hardware status | None | JSON object with status, device (`cuda`/`cpu`), and 6 class names | `200 OK` |
| `POST` | `/api/detect` | Upload vehicle image & run inference | `multipart/form-data` with `file: UploadFile` | Structured JSON containing damage status, metrics, detections array, and annotated image URL | `200 OK`<br>`400 Bad Request`<br>`500 Internal Error` |
| `GET` | `/api/results/{filename}` | Retrieve generated annotated image | Path parameter `filename` (e.g. `annotated_1f025746aec8.jpg`) | Binary image stream (`image/jpeg`, `image/png`, etc.) | `200 OK`<br>`400 Bad Request`<br>`404 Not Found` |

#### Sample `/api/health` Response
```json
{
  "status": "healthy",
  "service": "vehicle-damage-detection",
  "version": "1.0.0",
  "device": "cpu",
  "classes": [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat"
  ]
}
```

#### Sample `/api/detect` Response
```json
{
  "damage_status": "Damage Detected",
  "has_damage": true,
  "total_detections": 2,
  "unique_categories": [
    "dent",
    "tire flat"
  ],
  "average_confidence": 0.8351,
  "damage_area_percentage": 3.84,
  "detections": [
    {
      "class_id": 5,
      "class_name": "tire flat",
      "confidence": 0.8924,
      "x1": 142.5,
      "y1": 310.2,
      "x2": 268.0,
      "y2": 450.8,
      "bounding_box_area": 17646.3,
      "image_width": 640,
      "image_height": 480,
      "normalized_box_area": 0.0574
    }
  ],
  "image_width": 640,
  "image_height": 480,
  "inference_time_ms": 78.4,
  "annotated_image": "/api/results/annotated_1f025746aec8.jpg",
  "annotated_image_path": "C:\\Users\\rudra\\Desktop\\DLL Project\\outputs\\annotated_1f025746aec8.jpg"
}
```

---

## 9. DATABASE & STORAGE AUDIT

- **Database Engine:** **None.** The project currently has no relational or NoSQL database (no SQLite, PostgreSQL, MongoDB, or Redis).
- **Session State:** Stateless. Each request is independently evaluated without user session tracking.
- **Local Storage Architecture:**
  - `outputs/`: Primary filesystem cache storing generated annotated images (`annotated_<uuid>.<ext>`).
  - Temporary files: Uploaded images are written to the OS temp directory (`tempfile.gettempdir()`) with unique filenames and immediately deleted in a `finally:` block after OpenCV decoding.
  - Model weights: Persisted under `models/best.pt`.
  - Evaluation results: Persisted under `evaluation/detection/` (`results.json`, `results.csv`, `report.md`, and plot images).
- **Cloud Storage:** None currently configured (no AWS S3, Google Cloud Storage, or Azure Blob).

---

## 10. CONFIGURATION & ENVIRONMENT VARIABLES

- **Environment Files:** Neither `.env` nor `.env.example` currently exists in the workspace.
- **Active Configurations:** All settings are currently configured with default fallbacks in code:

| Setting / Constant | File Location | Default Fallback Value | Purpose |
| :--- | :--- | :--- | :--- |
| `API_BASE_URL` | `app/frontend/index.html` (L602) | `http://127.0.0.1:8000` | Backend API URL called by frontend JavaScript. |
| `ALLOWED_ORIGINS` | `app/main.py` (L53) | `localhost:3000`, `127.0.0.1:3000`, `5173` | Allowed CORS origins for browser access. |
| Model candidate paths | `src/detector.py` (L121) | `DataSet/models/best.pt`, `models/best.pt` | Search priority for loading weights. |
| `confidence_threshold` | `src/detector.py` (L85) | `0.25` | Default minimum confidence score for detection. |
| `iou_threshold` | `src/detector.py` (L86) | `0.45` | Non-Maximum Suppression (NMS) IoU threshold. |
| `output_dir` | `src/pipeline.py` (L99) | `outputs/` | Destination folder for saving annotated images. |
| Host & Port (API) | `scripts/run_api.py` (L25) | `127.0.0.1:8000` | Bind IP and port for FastAPI backend. |
| Host & Port (Frontend) | `scripts/run_frontend.py` (L18) | `127.0.0.1:3000` | Bind IP and port for static frontend server. |

---

## 11. DEPENDENCIES AUDIT

The project relies on `requirements.txt`:
```txt
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
pyyaml>=6.0
matplotlib>=3.7.0
pytest>=7.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
httpx>=0.24.0
```

### Dependency Analysis
- **Core ML & Vision:** `torch`, `torchvision`, `ultralytics`, `opencv-python`, `pillow`, `numpy`. All installed and operational.
- **Reporting & Evaluation:** `matplotlib`, `pyyaml`, `pandas` (pulled by Ultralytics).
- **Backend Service:** `fastapi`, `uvicorn`, `python-multipart`, `httpx`.
- **Test Framework:** `pytest`.
- **Potentially Missing in `requirements.txt`:** `pandas` is imported directly in `src/evaluation.py` (L15: `import pandas as pd`). Although installed in the current environment via Ultralytics transitive dependencies, it should be explicitly listed in `requirements.txt` to prevent installation errors in clean environments.

---

## 12. HOW TO RUN THE PROJECT

All instructions have been tested and verified on Windows with Python 3.11.

### 12.1 Prerequisites
- Python 3.10 or 3.11 installed.
- (Optional) NVIDIA CUDA-compatible GPU and drivers for GPU inference.

### 12.2 Installation
In the project root (`c:\Users\rudra\Desktop\DLL Project`):
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 12.3 Running the Automated Test Suite
To run all 31 unit and integration tests:
```bash
pytest tests/ -v
```
*(Note: Always specify `tests/` explicitly to avoid module collision with `scripts/`, as explained in Section 14).*

### 12.4 Running the FastAPI Backend Server
Start the backend API on `http://127.0.0.1:8000`:
```bash
python scripts/run_api.py
```
- Interactive Swagger API Documentation: `http://127.0.0.1:8000/docs`
- Redoc Documentation: `http://127.0.0.1:8000/redoc`
- Health check: `http://127.0.0.1:8000/api/health`

### 12.5 Running the Web Application Frontend
In a separate terminal, start the static HTTP server on `http://127.0.0.1:3000`:
```bash
python scripts/run_frontend.py
```
Then open a web browser and navigate to:
```
http://127.0.0.1:3000
```

### 12.6 Running Standalone Model Verification & Inference
To verify device detection, model loading, and execute a sample prediction:
```bash
python src/model_loader.py
```
Or run the dedicated test runners:
```bash
python scripts/test_detector.py
python scripts/test_pipeline.py
```

### 12.7 Running the Formal Evaluation Suite
To execute full evaluation on the 374-image test set and generate reports and charts:
```bash
python scripts/evaluate_model.py
```
Generated artifacts will be placed in `evaluation/detection/` (`results.json`, `results.csv`, `report.md`, and `plots/`).

---

## 13. CURRENT IMPLEMENTATION STATUS

| Feature / Component | Status | Evidence / File Path | Notes |
| :--- | :--- | :--- | :--- |
| **Trained YOLOv8 Weights** | COMPLETE | `models/best.pt` | 3.01M params, verified SHA-256 match with backup. |
| **Model Loader Module** | COMPLETE | `src/model_loader.py` | CUDA/CPU detection, weights locator, sample runner. |
| **Vehicle Damage Detector** | COMPLETE | `src/detector.py` | In-memory YOLO wrapper, confidence & IoU thresholding. |
| **End-to-End Pipeline** | COMPLETE | `src/pipeline.py` | Area calculation, class coloring, bounding boxes. |
| **Evaluation Suite** | COMPLETE | `src/evaluation.py` | Generates PR/F1 curves, CSV, JSON, Markdown report. |
| **FastAPI REST API** | COMPLETE | `app/main.py` | Health, detect upload, and static result serving. |
| **Frontend Web App** | COMPLETE | `app/frontend/index.html` | Drag & drop, live scanning animation, metrics, zoom modal. |
| **API Test Suite** | COMPLETE | `tests/test_api.py` | 8 tests covering happy paths, errors, and weight safety. |
| **Detector Test Suite** | COMPLETE | `tests/test_detector.py` | 7 tests covering bounding boxes, formats, empty detections. |
| **Pipeline Test Suite** | COMPLETE | `tests/test_pipeline.py` | 8 tests covering annotations, OpenCV rendering. |
| **Static HTML Test Suite**| COMPLETE | `tests/test_frontend_static.py` | 1 test validating all DOM IDs, elements, and claims. |
| **Evaluation Test Suite** | COMPLETE | `tests/test_evaluation.py` | 3 tests validating YAML generation and dataclasses. |
| **Evaluation Script** | COMPLETE | `scripts/evaluate_model.py` | CLI tool with argparse arguments. |
| **Backend Runner** | COMPLETE | `scripts/run_api.py` | Uvicorn runner with auto-reload. |
| **Frontend Runner** | COMPLETE | `scripts/run_frontend.py` | ThreadingHTTPServer serving on port 3000. |
| **Bare Pytest Execution** | BROKEN | `pytest` in root | Module collision between `scripts/` and `tests/`. Fixed by `pytest tests/`. |
| **Docker Containerization**| NOT IMPLEMENTED | Workspace root | No `Dockerfile` or `docker-compose.yml` exists. |
| **Persistent Database** | NOT IMPLEMENTED | Workspace root | Stateless architecture; no DB integration. |
| **Cloud Storage for Images**| NOT IMPLEMENTED | Workspace root | Images saved locally in `outputs/`. |
| **Automatic Cache Purge** | NOT IMPLEMENTED | `outputs/` | Output images accumulate on disk indefinitely. |
| **User Authentication** | NOT IMPLEMENTED | `app/main.py` | No JWT, OAuth, or API key authentication. |

---

## 14. KNOWN BUGS, ISSUES & GOTCHAS

### Confirmed Issues

1. **Pytest Module Name Collision (Root Directory Collection Error):**
   - **Problem:** Running bare `pytest` in the project root fails during test collection with:
     ```
     import file mismatch: imported module 'test_detector' has this __file__ attribute:
     C:\Users\rudra\Desktop\DLL Project\scripts\test_detector.py
     which is not the same as the test file we want to collect:
     C:\Users\rudra\Desktop\DLL Project\tests\test_detector.py
     ```
   - **Root Cause:** Both `scripts/` and `tests/` contain files named `test_detector.py` and `test_pipeline.py`. Because the filenames begin with `test_`, Pytest attempts to collect both as test modules, creating a Python module namespace conflict.
   - **Workaround:** Run `pytest tests/ -v` instead of bare `pytest`.
   - **Recommended Fix:** Rename `scripts/test_detector.py` to `scripts/run_detector_demo.py` and `scripts/test_pipeline.py` to `scripts/run_pipeline_demo.py`.

2. **Stale Kaggle Paths in `DataSEt/data.yaml`:**
   - **Problem:** `DataSEt/data.yaml` still contains the original training paths from Kaggle:
     ```yaml
     train: /kaggle/input/cardd-with-yolo-annotations-images-labels/train/images
     val: /kaggle/input/cardd-with-yolo-annotations-images-labels/val/images
     test: /kaggle/input/cardd-with-yolo-annotations-images-labels/test/images
     ```
   - **Impact:** Any attempt to train or evaluate using `yolo val data=DataSEt/data.yaml` fails on local machines.
   - **Current Workaround:** `src/evaluation.py` generates its own local POSIX config at `evaluation/detection/eval_dataset.yaml` with correct local paths.
   - **Recommended Fix:** Update `DataSEt/data.yaml` to use relative paths (`../DataSet/train/images`, etc.).

3. **Inconsistent Path Casing (`DataSEt` vs `DataSet`):**
   - **Problem:** On disk, the directory is named `DataSEt` (capital E). In various source files (`README.md`, `src/model_loader.py`, `src/detector.py`), it is referenced as `DataSet`.
   - **Impact:** While Windows file systems are case-insensitive and resolve the path correctly, deploying to Linux or inside Docker containers will cause `FileNotFoundError`.
   - **Recommended Fix:** Standardize directory name to lowercase `dataset/` or title-case `DataSet/` across all code and file structures.

4. **Missing Explicit `pandas` in `requirements.txt`:**
   - **Problem:** `src/evaluation.py` imports `pandas` directly (`import pandas as pd`), but `pandas` is not declared in `requirements.txt`.
   - **Impact:** While it works when `ultralytics` pulls `pandas` transitively, an environment installing minimal requirements could fail with `ModuleNotFoundError: No module named 'pandas'`.

5. **Indefinite Accumulation in `outputs/`:**
   - **Problem:** Every call to `POST /api/detect` writes an annotated file `annotated_<uuid>.<ext>` to `outputs/`.
   - **Impact:** Over time, high request volumes will consume disk space with no TTL or pruning mechanism.

---

## 15. CODE QUALITY AUDIT

- **Modularity:** High. `VehicleDamageDetector` is cleanly separated from `VehicleDamagePipeline`, which is separated from the FastAPI controller in `app/main.py`.
- **Type Annotations & Dataclasses:** Excellent. `Detection`, `DetectionResult`, `PipelineResult`, `ClassMetric`, and `EvaluationMetrics` all use Python `@dataclass` with clear type annotations (`dataclasses.asdict` used for serialization).
- **Error Handling:** Strong in `app/main.py`—handles missing files, unsupported extensions, empty uploads, and un-decodable images.
- **Resource Management:** Temporary files are safely cleaned up using `finally: temp_path.unlink()`.
- **Hardcoding:** Server addresses (`127.0.0.1:8000`, `127.0.0.1:3000`) and CORS origins are hardcoded rather than read from environment variables.
- **Test Coverage:** Extensive—31 tests covering unit, pipeline, API, static HTML validation, and model loading.

---

## 16. SECURITY REVIEW

### Verified Security Controls
1. **Path Traversal Protection:**
   In `app/main.py` (`/api/results/{filename}`), path traversal attacks (such as `../../windows/win.ini`) are prevented:
   ```python
   safe_name = Path(filename).name
   file_path = (pipeline.output_dir / safe_name).resolve()
   file_path.relative_to(pipeline.output_dir.resolve()) # Raises ValueError if traversed outside
   ```
2. **File Validation:**
   The upload handler validates file extensions against `SUPPORTED_EXTENSIONS`, rejects 0-byte files, and uses OpenCV (`cv2.imread`) to verify that the file is a genuine, decodable image rather than an arbitrary binary payload.
3. **Temporary File Security:**
   Uploaded files use `uuid.uuid4().hex[:12]` inside the system temp directory, preventing filename collision or overwrite attacks.

### Recommendations for Future Development
- **File Size Limit Middleware:** Add request body size limits in FastAPI/Uvicorn (e.g., max 25MB) before buffering large payloads into memory.
- **Rate Limiting:** Add `slowapi` or redis-based rate limiting to protect `/api/detect` from denial-of-service attempts.
- **CORS Hardening:** Move `ALLOWED_ORIGINS` to an environment variable for production environments.

---

## 17. PERFORMANCE PROFILE

- **Inference Latency:**
  - **CPU (Intel Core / AMD x86_64):** ~60 ms to 150 ms per image.
  - **GPU (NVIDIA CUDA):** ~10 ms to 25 ms per image.
- **Model Memory Footprint:**
  - In-memory footprint of `yolov8n.pt` is very low (~30 MB to 60 MB RAM).
- **Singleton Pattern:**
  - The model weights are loaded once at startup via the FastAPI `lifespan` handler and held in memory. Inference requests do not reload the model from disk.
- **Bottlenecks:**
  - Disk I/O: OpenCV reads and writes each annotated image to disk (`outputs/`). For high throughput, returning base64 encoded images or an in-memory buffer (`StreamingResponse(io.BytesIO(...))`) would reduce disk latency.

---

## 18. DEPLOYMENT & PRODUCTION READINESS

- **Current Deployment Status:** Configured strictly for local execution (`127.0.0.1`).
- **Containerization:** Not yet containerized (needs `Dockerfile` and `docker-compose.yml`).
- **Production Server:** Currently runs via `uvicorn.run(..., reload=True)`. Production deployments should use Gunicorn/Uvicorn workers (`gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker`).
- **Static Hosting:** In production, `app/frontend/index.html` should be served by NGINX, Cloudflare Pages, or FastAPI's `StaticFiles` mount rather than Python's `ThreadingHTTPServer`.

---

## 19. PROJECT HISTORY & CURRENT STATE

### Already Completed
- YOLOv8n model trained on CarDD dataset across 6 damage classes for 50 epochs.
- Weights verified and stored at `models/best.pt` with SHA-256 integrity verification.
- Test set evaluation script (`src/evaluation.py` and `scripts/evaluate_model.py`) reproducing Colab benchmarks (Precision 0.752, Recall 0.686, mAP@0.5 0.7214).
- Core detection module (`src/detector.py`) with dataclasses and confidence filtering.
- High-level analytics pipeline (`src/pipeline.py`) computing damage area % and rendering colored bounding boxes.
- FastAPI REST API (`app/main.py`) with `/api/health`, `/api/detect`, and `/api/results/{filename}`.
- Single-page web application (`app/frontend/index.html`) with drag-and-drop, scanning animation, results viewer, zoom, modal, and JSON export.
- 31 unit and integration tests across 6 test modules in `tests/`.

### Currently Working
- Full local end-to-end pipeline: running `run_api.py` and `run_frontend.py` provides a complete, interactive damage detection tool.

### Incomplete / Needs Fixing
- Module name collision between `scripts/test_*.py` and `tests/test_*.py` affecting root `pytest`.
- Outdated Kaggle paths in `DataSEt/data.yaml`.
- Case sensitivity in folder name (`DataSEt` vs `DataSet`).
- Missing explicit `pandas` in `requirements.txt`.
- No image purging or TTL for `outputs/`.
- No Docker containerization or cloud deployment configuration.

---

## 20. RECOMMENDED NEXT STEPS FOR CONTINUING DEVELOPMENT

### Priority 1 — Critical (Immediate Fixes)
1. **Fix Pytest Collection Collision:**
   - **Action:** Rename `scripts/test_detector.py` to `scripts/run_detector_demo.py` and `scripts/test_pipeline.py` to `scripts/run_pipeline_demo.py`.
   - **Reason:** Allows bare `pytest` to pass cleanly from the project root without import file mismatch errors.
2. **Standardize Dataset Folder Casing:**
   - **Action:** Rename folder `DataSEt` to `dataset` or `DataSet` and ensure all references across code match exactly.
   - **Reason:** Prevents `FileNotFoundError` on case-sensitive Linux/Docker environments.
3. **Update `DataSEt/data.yaml`:**
   - **Action:** Replace Kaggle paths with relative local paths.
   - **Reason:** Allows standard Ultralytics CLI commands (`yolo val data=...`) to execute without custom wrapper scripts.
4. **Add `pandas` to `requirements.txt`:**
   - **Action:** Add `pandas>=2.0.0` to `requirements.txt`.

### Priority 2 — Important (Production Readiness)
1. **Create Dockerfile & docker-compose:**
   - Provide a multi-stage Docker build packaging Python 3.11, CUDA/CPU PyTorch, Uvicorn, and NGINX for the frontend.
2. **Mount Static Frontend in FastAPI:**
   - Mount `app/frontend/` using FastAPI `StaticFiles`:
     ```python
     app.mount("/", StaticFiles(directory="app/frontend", html=True), name="frontend")
     ```
   - This eliminates the need to run two separate servers (`run_api.py` and `run_frontend.py`) by serving both API and UI on a single port.
3. **Add Environment Variable Configuration:**
   - Use `pydantic-settings` to manage `API_HOST`, `API_PORT`, `CORS_ORIGINS`, `CONFIDENCE_THRESHOLD`, and `MODEL_PATH`.

### Priority 3 — Enhancements (Product Features)
1. **Automated Pruning for `outputs/`:**
   - Add a background task or startup routine to delete annotated images older than 24 hours.
2. **In-Memory Image Streaming Option:**
   - Add a query parameter `?return_image=true` or return base64 encoded data to avoid disk I/O bottlenecks.
3. **Multi-Image Batch Inspection:**
   - Support uploading multiple photos of a vehicle (front, rear, side) and aggregating damage into a single report.

---

## 21. IMPORTANT FILES MAP

| File Path | Role / Function | Key Classes / Functions |
| :--- | :--- | :--- |
| `models/best.pt` | Trained model weights | YOLOv8n weights checkpoint (3.01M params) |
| `src/model_loader.py` | Model discovery & hardware verification | `detect_device()`, `locate_model()`, `load_model()` |
| `src/detector.py` | In-memory object detector wrapper | `VehicleDamageDetector`, `Detection`, `DetectionResult` |
| `src/pipeline.py` | Analytics, area calculation & annotation | `VehicleDamagePipeline`, `PipelineResult`, `generate_annotated_image()` |
| `src/evaluation.py` | Test set evaluation & metric reporting | `ModelEvaluator`, `EvaluationMetrics`, `run_evaluation()` |
| `app/main.py` | FastAPI backend REST API | `app`, `lifespan()`, `health_check()`, `detect_damage()`, `get_result_image()` |
| `app/frontend/index.html`| Full SPA UI application | Drag & drop, preview, scan animation, results viewer, zoom modal |
| `scripts/run_api.py` | Starts backend on port 8000 | `uvicorn.run("app.main:app", port=8000)` |
| `scripts/run_frontend.py`| Starts frontend on port 3000 | `ThreadingHTTPServer(..., port=3000)` |
| `scripts/evaluate_model.py`| CLI test set evaluation runner | Standalone evaluation script with argparse |
| `tests/test_api.py` | Pytest suite for FastAPI backend | Health check, file upload tests, security assertions |
| `tests/test_detector.py` | Pytest suite for detector module | Bounding box math, model persistence |
| `tests/test_pipeline.py` | Pytest suite for inference pipeline | OpenCV rendering, output structure |
| `DataSEt/data.yaml` | Dataset configuration | Class names & split paths (contains legacy Kaggle paths) |
| `evaluation/detection/results.json`| Benchmark test results | mAP@0.5, Precision, Recall, per-class F1 metrics |
| `requirements.txt` | Python dependency manifest | PyTorch, Ultralytics, OpenCV, FastAPI, etc. |

---

## 22. INSTRUCTIONS FOR THE NEXT AI ASSISTANT

### Context for Continuation
You are taking over development of the **Vehicle Damage Detection System**. 
- The project is fully functional locally.
- Do **NOT** retrain the model: the weights in `models/best.pt` are verified and achieve 0.7214 mAP@0.5 and 0.7173 F1 score across 6 classes (`dent`, `scratch`, `crack`, `glass shatter`, `lamp broken`, `tire flat`).
- Do **NOT** break existing tests: all 31 tests in `tests/` pass with `pytest tests/ -v`.
- The user runs Windows, but any containerization or deployment code must run cross-platform (mind the `DataSEt` directory name casing).

### Priority Tasks to Tackle First
1. **Pytest Cleanup:** Rename `scripts/test_detector.py` and `scripts/test_pipeline.py` so they don't begin with `test_` (e.g. `scripts/run_detector_demo.py` and `scripts/run_pipeline_demo.py`). Verify that bare `pytest` runs without collection errors.
2. **Single Server Architecture:** Mount `app/frontend` into `app/main.py` using `fastapi.staticfiles.StaticFiles` so the entire system runs on a single port (`8000`), simplifying deployment and eliminating CORS requirements.
3. **Containerization:** Create a clean `Dockerfile` and `docker-compose.yml` that bundles the application for CPU and optional GPU execution.
4. **Environment Configuration:** Extract hardcoded values into a `.env` file and configure them using `pydantic-settings`.

---

## 23. FINAL PROJECT SNAPSHOT

- **PROJECT:** DLL Project — Vehicle Damage Scanner
- **PURPOSE:** Automated detection, classification, and damage area appraisal of vehicle exterior damage using deep learning.
- **CURRENT STATUS:** Highly Functional Core (Model, Pipeline, REST API, Web App, Evaluation, 31 Tests Passing).
- **FRONTEND:** Single-page application (`app/frontend/index.html`) using Tailwind CSS and Vanilla JS; drag-and-drop, scan animation, 1.3x zoom, fullscreen modal, JSON download.
- **BACKEND:** FastAPI (`app/main.py`) with Uvicorn; lifespan-managed singleton model in memory; endpoints for health, image detection upload, and annotated image retrieval.
- **DEEP LEARNING MODEL:** YOLOv8n (3.01M parameters, ~5.95 MB weights at `models/best.pt`); trained for 50 epochs on CarDD dataset; detects 6 damage classes (`dent`, `scratch`, `crack`, `glass shatter`, `lamp broken`, `tire flat`).
- **DATASET:** CarDD YOLO format (4,000 total images: 2,816 train, 810 val, 374 test; 8,740 damage instances).
- **DATABASE:** None (Stateless REST service; outputs cached in `outputs/`).
- **MAIN API:** `POST /api/detect` (accepts multipart image upload, returns detections, area %, and annotated image URL).
- **DEPLOYMENT:** Local execution on ports 8000 (API) and 3000 (Frontend); not yet containerized.
- **COMPLETED:** Model weights, detector, pipeline, evaluation suite, FastAPI backend, web frontend, test suite (31/31 passing).
- **INCOMPLETE:** Docker containerization, single-port frontend mounting, database/cloud storage, automatic image cleanup.
- **CRITICAL ISSUES:** Root `pytest` collection collision due to `scripts/test_*.py` naming; stale Kaggle paths in `DataSEt/data.yaml`; case-sensitivity risk with `DataSEt` folder on Linux.
- **NEXT RECOMMENDED TASK:** Rename scripts to resolve pytest collision, then mount frontend static files directly in FastAPI.
- **HOW TO RUN:**
  1. Test Suite: `pytest tests/ -v`
  2. Backend: `python scripts/run_api.py` (http://127.0.0.1:8000)
  3. Frontend: `python scripts/run_frontend.py` (http://127.0.0.1:3000)
  4. Evaluation: `python scripts/evaluate_model.py`
