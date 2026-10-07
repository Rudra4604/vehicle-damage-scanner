"""Unit tests for the evaluation module.
"""
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.evaluation import ModelEvaluator, EvaluationMetrics, ClassMetric


@pytest.fixture(scope="session")
def evaluator(tmp_path_factory) -> ModelEvaluator:
    """Fixture providing a ModelEvaluator instance targeting a temp output dir."""
    out_dir = tmp_path_factory.mktemp("eval_test")
    return ModelEvaluator(output_dir=out_dir)


def test_evaluator_init(evaluator: ModelEvaluator):
    """Verify paths and device initialization."""
    assert evaluator.model_path.exists()
    assert evaluator.dataset_dir.exists()
    assert evaluator.output_dir.exists()
    assert evaluator.plots_dir.exists()
    assert evaluator.device in ["cuda", "cpu"]


def test_prepare_dataset_yaml(evaluator: ModelEvaluator):
    """Verify that generated eval dataset yaml contains valid splits and class names."""
    yaml_path = evaluator._prepare_dataset_yaml()
    assert yaml_path.exists()
    content = yaml_path.read_text(encoding="utf-8")
    assert "test: test/images" in content
    assert "dent" in content
    assert "tire flat" in content


def test_metric_dataclass_serialization():
    """Verify EvaluationMetrics serialization to dictionary."""
    mock_class = ClassMetric(
        class_id=0,
        class_name="dent",
        precision=0.75,
        recall=0.70,
        f1=0.7241,
        map50=0.71,
        map50_95=0.55,
        instances=100,
    )
    mock_metrics = EvaluationMetrics(
        overall_precision=0.75,
        overall_recall=0.70,
        overall_f1_harmonic=0.7241,
        overall_f1_macro=0.7241,
        overall_map50=0.71,
        overall_map50_95=0.55,
        per_class=[mock_class],
        num_images=50,
        total_instances=100,
        device="cpu",
        operating_point_notes={"note": "test"},
    )
    d = mock_metrics.to_dict()
    assert "overall" in d
    assert d["overall"]["precision"] == 0.75
    assert d["overall"]["f1_harmonic"] == 0.7241
    assert len(d["per_class"]) == 1
    assert d["per_class"][0]["class_name"] == "dent"
