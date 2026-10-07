# DLL Project - Car Damage Detection (YOLOv8)

This repository contains a trained YOLOv8 model for automated car damage detection, classification, and damage appraisal across 6 damage categories, hosted via a unified FastAPI server and interactive web interface.

## Project Structure

```
DLL Project/
├── dataset/              # Local CarDD dataset split (excluded from GitHub; ~2.83 GB)
├── models/
│   └── best.pt           # Trained YOLOv8n model weights (included for inference)
├── src/                  # Core detection, pipeline, evaluation, and loader modules
│   ├── detector.py
│   ├── evaluation.py
│   ├── model_loader.py
│   └── pipeline.py
├── app/                  # Application backend and frontend
│   ├── frontend/         # Compiled React 19 SPA (dist/) and fallback index.html
│   └── main.py           # Unified FastAPI server hosting both API and Frontend
├── evaluation/           # Evaluation metrics, reports, plots, and benchmark configs
├── outputs/              # Saved inferences and prediction visualizations
├── scripts/              # Operational runners and demo CLI scripts
│   ├── run_api.py        # Starts unified application server (port 8000)
│   ├── run_detector_demo.py
│   ├── run_pipeline_demo.py
│   └── evaluate_model.py
├── tests/                # Automated test suite (Pytest) with dedicated fixtures
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

> **Note on Dataset & Weights:**
> - `models/best.pt` is the included trained model checkpoint; normal web application inference and automated tests do not require downloading the full dataset.
> - `dataset/` is intentionally excluded from GitHub due to its size (~2.83 GB). Model re-training and test set evaluation (`scripts/evaluate_model.py`) require the local dataset.

## Classes

The trained model identifies 6 car damage classes:
1. `dent`
2. `scratch`
3. `crack`
4. `glass shatter`
5. `lamp broken`
6. `tire flat`

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify that all unit and integration tests pass:
```bash
pytest
```

### 3. Start the Unified Application
Run the unified server (hosting both the Web UI and the REST API on port 8000):
```bash
python scripts/run_api.py
```

Then access the application in your browser:
- **Web Application:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Endpoint:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### 4. Standalone Model Verification
Run the model loader script to check GPU/CPU status, load weights, display classes, and run a sample test inference:
```bash
python src/model_loader.py
```
