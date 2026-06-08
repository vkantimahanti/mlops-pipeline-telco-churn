import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, FunctionTransformer
)
from sklearn.impute import SimpleImputer

def build_preprocessor(feature_config: dict, 
                       pipeline_config: dict) -> ColumnTransformer:
    """
    Builds sklearn ColumnTransformer from config.
    All encoding and scaling decisions come from feature_config.yaml.
    """
    # --- Read from feature_config ---
    numeric_cols    = feature_config["features"]["numeric"]["columns"]
    skewed_cols     = feature_config["features"]["numeric"]["skewed"]
    ohe_cols        = feature_config["features"]["categorical"]["low_cardinality"]["columns"]
    ord_cols        = feature_config["features"]["categorical"]["medium_cardinality"]["columns"]
    scaling_method  = feature_config["features"]["numeric"]["scaling"]

    # --- Read from pipeline_config ---
    num_strategy    = pipeline_config["preprocessing"]["imputation"]["numeric_strategy"]
    cat_strategy    = pipeline_config["preprocessing"]["imputation"]["categorical_strategy"]

    # --- Scaler selection from config ---
    scaler_map = {
        "StandardScaler": StandardScaler(),
        "RobustScaler":   RobustScaler(),
        "MinMaxScaler":   MinMaxScaler()
    }
    scaler = scaler_map.get(scaling_method, StandardScaler())

    # --- Defensive Structural Guardrail ---
    # Ensure all skewed columns actually exist in the numeric definition list
    for col in skewed_cols:
        if col not in numeric_cols:
            raise ValueError(f" Configuration Error: Skewed column '{col}' must be declared in numeric columns list.")

    # --- Numeric pipelines ---
    # Skewed columns: impute → log → scale
    skewed_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=num_strategy)),
        # set validate=False if passing pandas DataFrames to let numpy natively handle elements safely
        ("log",     FunctionTransformer(np.log1p, validate=False)), 
        ("scaler",  scaler)
    ])

    # Normal numeric columns: impute → scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=num_strategy)),
        ("scaler",  scaler)
    ])

    # --- Categorical pipelines ---
    ohe_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=cat_strategy, fill_value="UNKNOWN")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ord_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=cat_strategy, fill_value="UNKNOWN")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    # --- Isolate Non-skewed numeric columns safely ---
    normal_numeric_cols = [c for c in numeric_cols if c not in skewed_cols]

    # --- Assemble ColumnTransformer ---
    transformers_list = []
    
    # Only append processors if columns actually exist in config to prevent empty pipeline errors
    if skewed_cols:
        transformers_list.append(("skewed_num", skewed_pipeline, skewed_cols))
    if normal_numeric_cols:
        transformers_list.append(("num", numeric_pipeline, normal_numeric_cols))
    if ohe_cols:
        transformers_list.append(("ohe", ohe_pipeline, ohe_cols))
    if ord_cols:
        transformers_list.append(("ord", ord_pipeline, ord_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder="drop"   # drops identifiers like customerID automatically!
    )

    print(" Preprocessor successfully built with robust conditional guardrails.")
    return preprocessor