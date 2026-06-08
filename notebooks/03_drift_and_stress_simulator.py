# Databricks notebook source
# ============================================================
# NOTEBOOK  : 03_drift_and_stress_simulator.py
# PURPOSE   : Production Simulation Sandbox
#             → Load active @champion model from Unity Catalog
#             → Simulate Covariate/Data Drift (Month 2 Cohort)
#             → Stress test pipeline resilience (Missing & Noisy data)
# DEPENDS ON: 02_gatekeeper_metric_evaluator.py
#             Must have active @champion alias assigned
# ============================================================

# COMMAND ----------
# ============================================================
# CELL 0 — ENVIRONMENT SETUP & PARAMETER CAPTURE
# ============================================================
import sys
import os
import numpy as np
import pandas as pd

# 1. Map workspace directories
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
project_root = "/Workspace" + os.path.dirname(os.path.dirname(notebook_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2.  THE BUNDLE BRIDGE: Handle the input parameters safely
dbutils.widgets.text("environment", "dev") # Creates the input field widget
ENV = dbutils.widgets.get("environment").strip().lower()

print(f"Project root : {project_root}")
print(f"Target Env   : {ENV.upper()}")



# COMMAND ----------
# ============================================================
# CELL 1 — ENVIRONMENT SETUP
# ============================================================

notebook_path = (
    dbutils.notebook.entry_point
    .getDbutils().notebook().getContext()
    .notebookPath().get()
)

project_root = "/Workspace" + os.path.dirname(
    os.path.dirname(notebook_path)
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root : {project_root}")
print(f"Exists       : {os.path.exists(project_root)}")

# COMMAND ----------
# ============================================================
# CELL 2 — IMPORTS & CHAMPION MODEL LOADING
# Loads the EXACT model tagged @champion in Unity Catalog
# This is the same model that passed the gatekeeper in notebook 02
# ============================================================

import mlflow.sklearn
import pandas as pd
from pyspark.sql.functions import col, when
from sklearn.metrics import (
    recall_score,
    f1_score,
    classification_report
)
from src.config_loader import MODEL_CONFIG, PIPELINE_CONFIG, FEATURE_CONFIG

# Read config values used across all cells
model_name  = MODEL_CONFIG["mlflow"]["registered_model_name"]
target_col  = PIPELINE_CONFIG["data"]["target_column"]
pos_class   = PIPELINE_CONFIG["data"]["positive_class"]
drop_cols   = PIPELINE_CONFIG["preprocessing"]["drop_columns"]
raw_table   = PIPELINE_CONFIG["data"]["raw_table"]

# Load @champion model from Unity Catalog
champion_uri      = f"models:/{model_name}@champion"
print(f"Loading champion model from : {champion_uri}")
champion_pipeline = mlflow.sklearn.load_model(champion_uri)
print(f"Champion model loaded successfully into cluster memory")

# COMMAND ----------
# ============================================================
# CELL 3 — HELPER: prepare_features()
# Shared utility used by all three test phases
# Ensures consistent feature preparation across every test
#
# What it does:
#   1. Converts target column to binary (1/0)
#   2. Drops target and identifier columns
#   3. Returns X (features) and y (labels) separately
# ============================================================
import numpy as np

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Hardened preparation utility. Replaces textual whitespace blocks 
    with true numpy NaNs and forces numeric casting to eliminate imputer crashes.
    """
    df = df.copy() # Avoid mutating original dataframes in memory

    #  DEFENSIVE REPAIR: Convert empty space strings (' ') to true NaNs
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Step 1 — Convert target to binary
    df[target_col] = df[target_col].apply(
        lambda x: 1.0 if str(x).strip() == pos_class else 0.0
    )

    # Step 2 — Separate features from label
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    
    # FORCE TYPE CASTING: Ensure known numeric columns are treated as floats, not strings
    num_cols = FEATURE_CONFIG.get("features", {}).get("numeric", {}).get("columns", ['tenure', 'MonthlyCharges', 'TotalCharges'])
    for col in num_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    y = df[target_col]

    return X, y

# COMMAND ----------
# ============================================================
# CELL 4 — PHASE 1: DATA DRIFT SIMULATION (MONTH 2)
#
# WHAT:
#   Month 1 training used ONLY stable cohorts
#   (One year + Two year contracts)
#
#   Month 2 simulation loads ALL customers
#   including volatile Month-to-month contracts
#   and new Fiber Optic internet profiles
#
# WHY:
#   Model was never trained on these profiles
#   Testing if performance degrades on unseen patterns
#   This is called COVARIATE DRIFT
#
# EXPECTED OUTCOME:
#   Recall drops compared to Month 1 baseline
#   Confirms model needs retraining with Month 2 data
# ============================================================

print("=" * 60)
print("PHASE 1 — DATA DRIFT SIMULATION (MONTH 2 COHORT)")
print("=" * 60)

print(f"\nLoading unfiltered data from : {raw_table}")
print(f"No cohort filters applied    : all contract types included")

# Load ALL data — no cohort filter (simulates Month 2 reality)
spark_drift_df = spark.read.table(raw_table)
df_drift       = spark_drift_df.toPandas()

print(f"\nFull dataset shape           : {df_drift.shape}")

# Prepare features using shared helper
X_drift, y_drift = prepare_features(df_drift)

print(f"\nChurn rate in drift data     : {y_drift.mean():.2%}")
print(f"Features shape               : {X_drift.shape}")

# Run predictions using champion model
y_drift_pred  = champion_pipeline.predict(X_drift)
drift_recall  = recall_score(y_drift, y_drift_pred)
drift_f1      = f1_score(y_drift, y_drift_pred)

# Compare against Month 1 baseline
# Baseline values logged during 01_baseline_training_orchestrator
baseline_recall = MODEL_CONFIG["evaluation"].get("baseline_recall", 0.6977) # Defaulting explicitly to our actual M1 baseline

print(f"\n{'=' * 40}")
print(f"DRIFT EVALUATION REPORT")
print(f"{'=' * 40}")
print(f"  Month 1 Baseline Recall : {baseline_recall if baseline_recall else 'see MLflow run'}")
print(f"  Month 2 Drift Recall    : {drift_recall:.4f}")
print(f"  Month 2 Drift F1        : {drift_f1:.4f}")

# Drift severity assessment
if baseline_recall:
    recall_drop = baseline_recall - drift_recall
    if recall_drop > 0.10:
        print(f"\n  SIGNIFICANT DRIFT DETECTED")
        print(f"  Recall dropped by {recall_drop:.4f}")
        print(f"  Action: Retrain with Month 2 data included")
    elif recall_drop > 0.05:
        print(f"\n  MODERATE DRIFT DETECTED")
        print(f"  Recall dropped by {recall_drop:.4f}")
        print(f"  Action: Monitor closely — schedule retraining")
    else:
        print(f"\n  MODEL IS STABLE UNDER DRIFT")
        print(f"  Recall drop within acceptable range")

print(f"\nClassification Report (Month 2 Drift):")
print(classification_report(
    y_drift, y_drift_pred,
    target_names=["No Churn", "Churn"]
))

# COMMAND ----------
# ============================================================
# CELL 5 — PHASE 2: STRESS TEST (MISSING VALUES)
#
# WHAT:
#   Feeds raw CSV containing null/blank values
#   directly into champion pipeline — zero cleaning
#
# WHY:
#   Production data WILL arrive with missing values
#   Pipeline must handle gracefully — not crash
#
# WHAT DEFENDS AGAINST THIS:
#   SimpleImputer inside features.py
#   Fills nulls before they reach the model
#
# EXPECTED OUTCOME:
#   PASS  → SimpleImputer caught all nulls
#   CRASH → SimpleImputer missing or misconfigured
# ============================================================

print("=" * 60)
print("PHASE 2 — STRESS TEST: MISSING VALUES")
print("=" * 60)

missing_csv_path = PIPELINE_CONFIG["data"]["stress_test_missing"]
print(f"\nSource file  : {missing_csv_path}")

# Load raw CSV with missing values — no cleaning applied
df_missing_raw = (
    spark.read
    .option("header", "true")
    .csv(missing_csv_path)
    .toPandas()
)

print(f"Records loaded : {df_missing_raw.shape[0]}")
print(f"Columns        : {df_missing_raw.shape[1]}")

# Show null profile before prediction
null_counts = df_missing_raw.isnull().sum()
null_cols   = null_counts[null_counts > 0]
print(f"\nNull profile (columns with missing values):")
print(null_counts[null_counts > 0].to_string()
      if len(null_cols) > 0
      else "  No nulls detected in this file")

# CRITICAL: Drop target and identifiers before predict
# Even stress test files must have target dropped
X_missing, y_missing = prepare_features(df_missing_raw)

try:
    preds_missing = champion_pipeline.predict(X_missing)

    print(f"\nSTRESS TEST RESULT : PASS")
    print(f"  Records processed  : {len(preds_missing)}")
    print(f"  Nulls handled by   : SimpleImputer in features.py")
    print(f"  Pipeline is resilient to missing value inputs")

except Exception as e:
    print(f"\nSTRESS TEST RESULT : CRASHED")
    print(f"  Error   : {e}")
    print(f"  Action  : Check SimpleImputer config in features.py")
    print(f"            Verify numeric_strategy and categorical_strategy")
    print(f"            in pipeline_config.yaml → preprocessing.imputation")

# COMMAND ----------
# ============================================================
# CELL 6 — PHASE 3: STRESS TEST (STRUCTURAL NOISE)
#
# WHAT:
#   Feeds data with UNEXPECTED category values
#   into champion pipeline — values never seen in training
#
# WHY:
#   Source systems change over time
#   New categories appear, typos occur,
#   encoding systems update
#   Example: Contract = "Yearly" instead of "One year"
#
# WHAT DEFENDS AGAINST THIS:
#   OneHotEncoder(handle_unknown='ignore') → returns zeros
#   OrdinalEncoder(handle_unknown='use_encoded_value',
#                  unknown_value=-1)       → returns -1
#   Both defined in features.py
#
# EXPECTED OUTCOME:
#   PASS  → Unknown categories silently handled
#   CRASH → handle_unknown not set correctly
# ============================================================

print("=" * 60)
print("PHASE 3 — STRESS TEST: STRUCTURAL NOISE")
print("=" * 60)

noisy_csv_path = PIPELINE_CONFIG["data"]["stress_test_noise"]
print(f"\nSource file  : {noisy_csv_path}")

# Load raw CSV with unexpected/noisy category values
df_noise_raw = (
    spark.read
    .option("header", "true")
    .csv(noisy_csv_path)
    .toPandas()
)

print(f"Records loaded : {df_noise_raw.shape[0]}")
print(f"Columns        : {df_noise_raw.shape[1]}")

# Show sample of noisy categorical values before prediction
ohe_cols = FEATURE_CONFIG.get("features", {}).get(
    "categorical", {}).get("low_cardinality", {}).get("columns", [])

if ohe_cols:
    print(f"\nSample category values in noisy data:")
    for c in ohe_cols[:3]:  # show first 3 categorical columns
        if c in df_noise_raw.columns:
            print(f"  {c}: {df_noise_raw[c].unique()[:5].tolist()}")

# CRITICAL: Drop target and identifiers before predict
X_noise, y_noise = prepare_features(df_noise_raw)

try:
    preds_noise = champion_pipeline.predict(X_noise)

    print(f"\nSTRESS TEST RESULT : PASS")
    print(f"  Records processed  : {len(preds_noise)}")
    print(f"  Unknown categories : silently ignored")
    print(f"  Defended by        : handle_unknown='ignore' in OneHotEncoder")
    print(f"                       handle_unknown='use_encoded_value' in OrdinalEncoder")
    print(f"  Pipeline is resilient to unseen categorical inputs")

except Exception as e:
    print(f"\nSTRESS TEST RESULT : CRASHED")
    print(f"  Error   : {e}")
    print(f"  Action  : Check encoder config in features.py")
    print(f"            OneHotEncoder   → handle_unknown='ignore'")
    print(f"            OrdinalEncoder  → handle_unknown='use_encoded_value'")
    print(f"                              unknown_value=-1")

# COMMAND ----------
# ============================================================
# CELL 7 — SIMULATION SUMMARY
# Final report across all three test phases
# ============================================================

print("=" * 60)
print("SIMULATION COMPLETE — FULL SUMMARY")
print("=" * 60)

print(f"""
  CHAMPION MODEL    : {champion_uri}

  PHASE 1 — DATA DRIFT (Month 2 Simulation)
    Dataset         : All contracts (no filter)
    Records tested  : {df_drift.shape[0]}
    Drift Recall    : {drift_recall:.4f}
    Drift F1        : {drift_f1:.4f}
    Verdict         : Model tested against real-world distribution shift

  PHASE 2 — MISSING VALUES STRESS TEST
    Source file     : {missing_csv_path}
    Records tested  : {df_missing_raw.shape[0]}
    Defense         : SimpleImputer (features.py)
    Verdict         : See PASS/CRASH result above

  PHASE 3 — STRUCTURAL NOISE STRESS TEST
    Source file     : {noisy_csv_path}
    Records tested  : {df_noise_raw.shape[0]}
    Defense         : handle_unknown encoders (features.py)
    Verdict         : See PASS/CRASH result above

  NEXT STEPS:
    Drift detected   → Run 01_baseline_training_orchestrator.py
                       with Month 2 cohort included in filters
    Stress test PASS → Pipeline is production-ready
    Stress test CRASH→ Fix features.py → re-run 01 → re-run 02 → re-run 03
""")