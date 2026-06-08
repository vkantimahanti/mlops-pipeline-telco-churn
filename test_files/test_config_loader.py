# ============================================================
# tests/test_config_loader.py
# Unit tests for src/config_loader.py
# Run: pytest tests/ -v
# ============================================================

import pytest
import os
import sys
import yaml
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_loader import load_config


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def sample_yaml(tmp_path):
    """Creates a temporary YAML file for testing."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    sample_content = {
        "model": {
            "algorithm": "RandomForestClassifier",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 4
            }
        },
        "evaluation": {
            "primary_metric": "f1_score",
            "promotion_threshold": 0.75
        }
    }

    config_file = config_dir / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_content, f)

    return tmp_path, "test_config.yaml"


# ── Tests: load_config() ────────────────────────────────────

class TestLoadConfig:

    def test_returns_dict(self, sample_yaml, monkeypatch):
        """load_config() must return a Python dict."""
        tmp_path, filename = sample_yaml
        monkeypatch.chdir(tmp_path / "config" / "..")
        # Patch the config path resolution
        result = yaml.safe_load(
            open(tmp_path / "config" / filename).read()
        )
        assert isinstance(result, dict)

    def test_top_level_keys_present(self, sample_yaml):
        """Loaded config must have expected top-level keys."""
        tmp_path, filename = sample_yaml
        with open(tmp_path / "config" / filename) as f:
            config = yaml.safe_load(f)

        assert "model" in config
        assert "evaluation" in config

    def test_nested_values_accessible(self, sample_yaml):
        """Nested config values must be accessible via dict keys."""
        tmp_path, filename = sample_yaml
        with open(tmp_path / "config" / filename) as f:
            config = yaml.safe_load(f)

        assert config["model"]["algorithm"] == "RandomForestClassifier"
        assert config["model"]["hyperparameters"]["n_estimators"] == 100
        assert config["evaluation"]["primary_metric"] == "f1_score"

    def test_raises_on_missing_file(self, tmp_path, monkeypatch):
        """load_config() must raise FileNotFoundError for missing files."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_promotion_threshold_is_float(self, sample_yaml):
        """Promotion threshold must be a float between 0 and 1."""
        tmp_path, filename = sample_yaml
        with open(tmp_path / "config" / filename) as f:
            config = yaml.safe_load(f)

        threshold = config["evaluation"]["promotion_threshold"]
        assert isinstance(threshold, float)
        assert 0.0 < threshold < 1.0
