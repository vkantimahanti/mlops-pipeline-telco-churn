# ============================================================
# tests/test_features.py
# Unit tests for src/features.py
# Run: pytest tests/ -v
# ============================================================

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.features import build_preprocessor


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def mock_feature_config():
    """Minimal feature_config matching pipeline structure."""
    return {
        "features": {
            "numeric": {
                "columns": ["tenure", "MonthlyCharges", "TotalCharges"],
                "scaling": "StandardScaler",
                "skewed": ["TotalCharges"]
            },
            "categorical": {
                "low_cardinality": {
                    "columns": ["gender", "Partner"],
                    "encoder": "OneHotEncoder",
                    "max_unique_values": 10
                },
                "medium_cardinality": {
                    "columns": ["Contract", "InternetService"],
                    "encoder": "OrdinalEncoder",
                    "max_unique_values": 50
                }
            }
        }
    }


@pytest.fixture
def mock_pipeline_config():
    """Minimal pipeline_config matching preprocessing structure."""
    return {
        "preprocessing": {
            "imputation": {
                "numeric_strategy": "median",
                "categorical_strategy": "most_frequent"
            }
        }
    }


@pytest.fixture
def sample_dataframe():
    """Minimal dataframe matching feature config columns."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "tenure":         np.random.randint(0, 72, n),
        "MonthlyCharges": np.random.uniform(18, 118, n),
        "TotalCharges":   np.random.uniform(0, 8684, n),
        "gender":         np.random.choice(["Male", "Female"], n),
        "Partner":        np.random.choice(["Yes", "No"], n),
        "Contract":       np.random.choice(["One year", "Two year", "Month-to-month"], n),
        "InternetService":np.random.choice(["DSL", "Fiber optic", "No"], n),
    })


# ── Tests: build_preprocessor() ─────────────────────────────

class TestBuildPreprocessor:

    def test_returns_column_transformer(
        self, mock_feature_config, mock_pipeline_config
    ):
        """build_preprocessor() must return a ColumnTransformer."""
        preprocessor = build_preprocessor(
            mock_feature_config, mock_pipeline_config
        )
        assert isinstance(preprocessor, ColumnTransformer)

    def test_has_expected_transformer_names(
        self, mock_feature_config, mock_pipeline_config
    ):
        """ColumnTransformer must contain all expected sub-pipelines."""
        preprocessor = build_preprocessor(
            mock_feature_config, mock_pipeline_config
        )
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "skewed_num" in transformer_names
        assert "num" in transformer_names
        assert "ohe" in transformer_names
        assert "ord" in transformer_names

    def test_remainder_is_drop(
        self, mock_feature_config, mock_pipeline_config
    ):
        """Unlisted columns must be dropped — prevents identifier leakage."""
        preprocessor = build_preprocessor(
            mock_feature_config, mock_pipeline_config
        )
        assert preprocessor.remainder == "drop"

    def test_fit_transform_runs_without_error(
        self, mock_feature_config, mock_pipeline_config, sample_dataframe
    ):
        """Preprocessor must fit and transform sample data without crashing."""
        preprocessor = build_preprocessor(
            mock_feature_config, mock_pipeline_config
        )
        result = preprocessor.fit_transform(sample_dataframe)
        assert result is not None
        assert result.shape[0] == len(sample_dataframe)

    def test_handles_missing_values(
        self, mock_feature_config, mock_pipeline_config, sample_dataframe
    ):
        """Preprocessor must handle NaN values via SimpleImputer."""
        df_with_nulls = sample_dataframe.copy()
        # Inject nulls into numeric and categorical columns
        df_with_nulls.loc[0:5, "TotalCharges"] = np.nan
        df_with_nulls.loc[6:10, "gender"] = np.nan

        preprocessor = build_preprocessor(
            mock_feature_config, mock_pipeline_config
        )
        # Must not raise — SimpleImputer handles nulls
        result = preprocessor.fit_transform(df_with_nulls)
        assert not np.isnan(result).any()

    def test_handles_unseen_categories(
        self, mock_feature_config, mock_pipeline_config, sample_dataframe
    ):
        """Preprocessor must handle unseen categories without crashing."""
        preprocessor = build_preprocessor(
            mock_feature_config, mock_pipeline_config
        )
        preprocessor.fit(sample_dataframe)

        # Introduce category never seen in training
        df_unseen = sample_dataframe.copy()
        df_unseen.loc[0, "Contract"] = "UNKNOWN_CONTRACT_TYPE"
        df_unseen.loc[0, "gender"]   = "UNKNOWN_GENDER"

        # Must not raise — handle_unknown='ignore' / use_encoded_value
        result = preprocessor.transform(df_unseen)
        assert result is not None

    def test_output_has_no_nulls_after_transform(
        self, mock_feature_config, mock_pipeline_config, sample_dataframe
    ):
        """Transformed output must contain zero NaN values."""
        preprocessor = build_preprocessor(
            mock_feature_config, mock_pipeline_config
        )
        result = preprocessor.fit_transform(sample_dataframe)
        assert not np.isnan(result).any(), \
            "Transformed output contains NaN — check SimpleImputer config"

    def test_skewed_column_gets_log_transform(
        self, mock_feature_config, mock_pipeline_config
    ):
        """Skewed columns must pass through log transform step."""
        preprocessor = build_preprocessor(
            mock_feature_config, mock_pipeline_config
        )
        # Find the skewed_num transformer
        skewed_transformer = None
        for name, transformer, cols in preprocessor.transformers:
            if name == "skewed_num":
                skewed_transformer = transformer
                break

        assert skewed_transformer is not None
        step_names = [step[0] for step in skewed_transformer.steps]
        assert "log" in step_names, \
            "Skewed pipeline missing log transform step"
        assert "scaler" in step_names, \
            "Skewed pipeline missing scaler step"
        assert "imputer" in step_names, \
            "Skewed pipeline missing imputer step"
