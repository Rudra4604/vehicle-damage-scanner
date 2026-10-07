"""Unit and integration tests for centralized configuration and output file pruning.
"""
import os
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.config import Settings, settings
from src.pipeline import VehicleDamagePipeline, prune_output_directory
from src.detector import get_default_project_root


def test_settings_defaults():
    """Verify standard default configuration values."""
    assert settings.HOST == "127.0.0.1"
    assert settings.PORT == 8000
    assert settings.DEBUG is False
    assert settings.CONFIDENCE_THRESHOLD == 0.25
    assert settings.IOU_THRESHOLD == 0.45
    assert settings.OUTPUT_RETENTION_HOURS == 24
    assert settings.MAX_OUTPUT_FILES == 500
    assert settings.OUTPUT_DIR == Path("outputs")
    assert settings.OUTPUT_DIRECTORY == Path("outputs")
    assert isinstance(settings.ALLOWED_ORIGINS, list)
    assert "http://localhost:3000" in settings.ALLOWED_ORIGINS


def test_settings_environment_overrides(monkeypatch):
    """Verify that environment variables cleanly override settings."""
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.40")
    monkeypatch.setenv("IOU_THRESHOLD", "0.55")
    monkeypatch.setenv("OUTPUT_RETENTION_HOURS", "48")
    monkeypatch.setenv("MAX_OUTPUT_FILES", "250")
    monkeypatch.setenv("OUTPUT_DIRECTORY", "custom_outputs")
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "https://example.com, https://app.example.com",
    )

    custom_settings = Settings()
    assert custom_settings.HOST == "0.0.0.0"
    assert custom_settings.PORT == 9090
    assert custom_settings.DEBUG is True
    assert pytest.approx(custom_settings.CONFIDENCE_THRESHOLD, 0.01) == 0.40
    assert pytest.approx(custom_settings.IOU_THRESHOLD, 0.01) == 0.55
    assert custom_settings.OUTPUT_RETENTION_HOURS == 48
    assert custom_settings.MAX_OUTPUT_FILES == 250
    assert custom_settings.OUTPUT_DIR == Path("custom_outputs")
    assert custom_settings.OUTPUT_DIRECTORY == Path("custom_outputs")
    assert custom_settings.ALLOWED_ORIGINS == [
        "https://example.com",
        "https://app.example.com",
    ]


def test_prune_output_directory_missing_dir():
    """Verify safe handling of non-existent output directory."""
    non_existent = PROJECT_ROOT / "non_existent_output_dir_99999"
    assert not non_existent.exists()
    pruned = prune_output_directory(output_dir=non_existent)
    assert pruned == 0


def test_prune_output_directory_retention_hours(tmp_path: Path):
    """Verify files older than retention hours are deleted while newer files and .gitkeep are preserved."""
    test_dir = tmp_path / "test_outputs"
    test_dir.mkdir()

    old_file = test_dir / "annotated_old.jpg"
    old_file.write_text("old image dummy")
    new_file = test_dir / "annotated_new.jpg"
    new_file.write_text("new image dummy")
    gitkeep_file = test_dir / ".gitkeep"
    gitkeep_file.write_text("")

    now = time.time()
    # Set old_file mtime to 30 hours ago
    os.utime(old_file, (now - 30 * 3600, now - 30 * 3600))
    # Set gitkeep mtime to 40 hours ago (should still be preserved)
    os.utime(gitkeep_file, (now - 40 * 3600, now - 40 * 3600))
    # Set new_file mtime to current time
    os.utime(new_file, (now, now))

    pruned = prune_output_directory(
        output_dir=test_dir,
        retention_hours=24,
        max_files=100,
    )

    assert pruned == 1
    assert not old_file.exists(), "Old file should have been deleted"
    assert new_file.exists(), "New file should be preserved"
    assert gitkeep_file.exists(), ".gitkeep placeholder should never be deleted"


def test_prune_output_directory_max_files(tmp_path: Path):
    """Verify MAX_OUTPUT_FILES limit prunes oldest files and keeps newest."""
    test_dir = tmp_path / "test_outputs_max"
    test_dir.mkdir()

    now = time.time()
    created_files = []
    for i in range(5):
        f = test_dir / f"annotated_{i}.jpg"
        f.write_text(f"dummy {i}")
        # Offset mtime: i=0 is oldest, i=4 is newest
        mtime = now + (i * 10)
        os.utime(f, (mtime, mtime))
        created_files.append(f)

    # Allow retention_hours=100 (so no file is pruned by age), but max_files=3
    pruned = prune_output_directory(
        output_dir=test_dir,
        retention_hours=100,
        max_files=3,
    )

    assert pruned == 2
    # The 2 oldest (annotated_0 and annotated_1) should be deleted
    assert not created_files[0].exists()
    assert not created_files[1].exists()
    # The 3 newest (annotated_2, annotated_3, annotated_4) must remain
    assert created_files[2].exists()
    assert created_files[3].exists()
    assert created_files[4].exists()


def test_prune_output_directory_protected_directories():
    """Verify protected system/project directories are never touched by pruner."""
    root = get_default_project_root()
    models_dir = root / "models"
    dataset_dir = root / "dataset"
    src_dir = root / "src"

    assert prune_output_directory(output_dir=root) == 0
    assert prune_output_directory(output_dir=models_dir) == 0
    assert prune_output_directory(output_dir=dataset_dir) == 0
    assert prune_output_directory(output_dir=src_dir) == 0

    # Ensure best.pt is fully intact
    assert (models_dir / "best.pt").exists()


def test_pipeline_prune_outputs_method(tmp_path: Path):
    """Verify prune_outputs method on VehicleDamagePipeline operates correctly."""
    test_dir = tmp_path / "pipeline_outputs"
    test_dir.mkdir()

    old_file = test_dir / "annotated_pipeline_old.jpg"
    old_file.write_text("pipeline old dummy")
    now = time.time()
    os.utime(old_file, (now - 50 * 3600, now - 50 * 3600))

    pipeline = VehicleDamagePipeline(output_dir=test_dir)
    pruned = pipeline.prune_outputs(retention_hours=24, max_files=50)

    assert pruned == 1
    assert not old_file.exists()
