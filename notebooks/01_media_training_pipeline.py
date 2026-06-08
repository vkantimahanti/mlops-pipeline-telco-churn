# Databricks notebook source
# ============================================================
# NOTEBOOK  : 01_baseline_training_orchestrator.py
# PURPOSE   : Establish Month 1 baseline model
#             → Load stable cohort data from Delta table
#             → Preprocess via features.py
#             → Train Random Forest baseline
#             → Register Version 1 to Unity Catalog Model Registry
# TRIGGER   : Run ONCE to establish baseline
# DEPENDS ON: 00_data_ingestion.py must have run first
# ============================================================

# COMMAND ----------
# ============================================================
# CELL 0 — ENVIRONMENT SETUP & PARAMETER CAPTURE
# ============================================================
import sys
import os

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
# Resolve project root and register src/ on sys.path
# Must run before any src/ imports
# ============================================================
 
notebook_path = (
    dbutils.notebook.entry_point
    .getDbutils().notebook().getContext()
    .notebookPath().get()
)
 
# notebooks/ → project_root (one level up)
project_root = "/Workspace" + os.path.dirname(
    os.path.dirname(notebook_path)
)
 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
 
print(f"Project root : {project_root}")
print(f"sys.path[0]  : {sys.path[0]}")
print(f"Exists       : {os.path.exists(project_root)}")
 
# COMMAND ----------
# ============================================================
# CELL 2 — IMPORTS
# All src/ modules loaded after sys.path is confirmed
# ============================================================
 
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
 
from sklearn.ensemble        import RandomForestClassifier
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    f1_score, roc_auc_score,
    precision_score, recall_score,
    classification_report, confusion_matrix
)
 
# Modular src imports
from src.config_loader import MODEL_CONFIG, PIPELINE_CONFIG, FEATURE_CONFIG
from src.data_loader   import load_training_data
from src.features      import build_preprocessor
 
print("All imports successful")
 
# COMMAND ----------
# ============================================================
# CELL 3 — CONFIGURATION AUDIT
# Print resolved config values before any processing starts
# Confirms correct YAML files are loaded — no silent misconfig
# ============================================================
 
print("=" * 60)
print("CONFIGURATION AUDIT — BASELINE TRAINING")
print("=" * 60)
 
print("\n[ PIPELINE CONFIG ]")
print(f"  Raw table       : {PIPELINE_CONFIG['data']['raw_table']}")
print(f"  Target column   : {PIPELINE_CONFIG['data']['target_column']}")
print(f"  Positive class  : {PIPELINE_CONFIG['data']['positive_class']}")
print(f"  Drop columns    : {PIPELINE_CONFIG['preprocessing']['drop_columns']}")
print(f"  Contract cohorts: {PIPELINE_CONFIG['data']['filters']['contract_cohorts']}")
print(f"  Min tenure      : {PIPELINE_CONFIG['data']['filters']['min_tenure_months']} months")
 
print("\n[ MODEL CONFIG ]")
print(f"  Algorithm       : {MODEL_CONFIG['model']['algorithm']}")
print(f"  n_estimators    : {MODEL_CONFIG['model']['hyperparameters']['n_estimators']}")
print(f"  max_depth       : {MODEL_CONFIG['model']['hyperparameters']['max_depth']}")
print(f"  class_weight    : {MODEL_CONFIG['model']['class_weight']}")
print(f"  test_size       : {MODEL_CONFIG['training']['test_size']}")
print(f"  stratify        : {MODEL_CONFIG['training']['stratify']}")
print(f"  Primary metric  : {MODEL_CONFIG['evaluation']['primary_metric']}")
print(f"  Promo threshold : {MODEL_CONFIG['evaluation']['promotion_threshold']}")
 
print("\n[ MLFLOW CONFIG ]")
print(f"  Experiment      : {MODEL_CONFIG['mlflow']['experiment_path']}")
print(f"  Run name        : {MODEL_CONFIG['mlflow']['run_name']}")
print(f"  Registered name : {MODEL_CONFIG['mlflow']['registered_model_name']}")
 
print("\n[ FEATURE CONFIG ]")
print(f"  Numeric cols    : {FEATURE_CONFIG['features']['numeric']['columns']}")
print(f"  Skewed cols     : {FEATURE_CONFIG['features']['numeric']['skewed']}")
print(f"  OHE cols        : {FEATURE_CONFIG['features']['categorical']['low_cardinality']['columns']}")
print(f"  Ordinal cols    : {FEATURE_CONFIG['features']['categorical']['medium_cardinality']['columns']}")
 
# COMMAND ----------
# ============================================================
# CELL 4 — DATA LOADING
# Reads from registered Delta table
# Applies Month 1 cohort filter (stable contracts only)
# Converts target to binary
# ============================================================
 
print("=" * 60)
print("PHASE 1 — DATA LOADING (Month 1 Baseline Cohort)")
print("=" * 60)
 
df = load_training_data(spark, PIPELINE_CONFIG)
 
print(f"\nDataset shape       : {df.shape}")
print(f"Columns             : {list(df.columns)}")
 
# Target distribution audit
target_col = PIPELINE_CONFIG["data"]["target_column"]
target_dist = df[target_col].value_counts(normalize=True) * 100
 
print(f"\nTarget distribution ('{target_col}'):")
print(target_dist.to_string())
 
# Imbalance warning
majority_pct = target_dist.max()
if majority_pct > 60:
    print(f"\nIMBALANCE DETECTED: majority class = {majority_pct:.1f}%")
    print(f"class_weight='{MODEL_CONFIG['model']['class_weight']}' will compensate")
else:
    print(f"\nClasses are balanced — no special handling required")
 
# COMMAND ----------
# ============================================================
# CELL 5 — FEATURE PREPARATION
# Separate features (X) from target (y)
# Drop identifier columns defined in pipeline_config
# ============================================================
 
print("=" * 60)
print("PHASE 2 — FEATURE PREPARATION")
print("=" * 60)
 
drop_cols = PIPELINE_CONFIG["preprocessing"]["drop_columns"]
 
X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
y = df[target_col]
 
print(f"Features shape  : {X.shape}")
print(f"Target shape    : {y.shape}")
print(f"Dropped columns : {drop_cols}")
print(f"\nFeature columns :\n{list(X.columns)}")
 
# COMMAND ----------
# ============================================================
# CELL 6 — TRAIN / TEST SPLIT
# stratify=y preserves class ratio in both splits
# Ratio defined in model_config.yaml → training.test_size
# ============================================================
 
print("=" * 60)
print("PHASE 3 — TRAIN / TEST SPLIT")
print("=" * 60)
 
test_size    = MODEL_CONFIG["training"]["test_size"]
random_state = MODEL_CONFIG["model"]["random_state"]
stratify     = MODEL_CONFIG["training"]["stratify"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = test_size,
    random_state = random_state,
    stratify     = y if stratify else None
)
 
print(f"Train set : {X_train.shape} | Class ratio: {y_train.mean():.2%} positive")
print(f"Test set  : {X_test.shape}  | Class ratio: {y_test.mean():.2%} positive")
print(f"stratify  : {stratify} — class ratio preserved in both splits")
 
# COMMAND ----------
# ============================================================
# CELL 7 — BUILD PIPELINE
# Preprocessor built from feature_config + pipeline_config
# Combined with RandomForest into single sklearn Pipeline
# ============================================================
 
print("=" * 60)
print("PHASE 4 — BUILDING SKLEARN PIPELINE")
print("=" * 60)
 
# Preprocessor from features.py (config-driven)
preprocessor = build_preprocessor(FEATURE_CONFIG, PIPELINE_CONFIG)
 
# Full pipeline: preprocessing + model
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators = MODEL_CONFIG["model"]["hyperparameters"]["n_estimators"],
        max_depth    = MODEL_CONFIG["model"]["hyperparameters"]["max_depth"],
        class_weight = MODEL_CONFIG["model"]["class_weight"],
        random_state = MODEL_CONFIG["model"]["random_state"]
    ))
])
 
print("Pipeline steps:")
for step_name, step_obj in full_pipeline.steps:
    print(f"  [{step_name}] → {type(step_obj).__name__}")
 
# COMMAND ----------
# ============================================================
# CELL 8 — TRAINING + MLFLOW TRACKING
# Trains pipeline on training set
# Evaluates on held-out test set
# Logs all params, metrics, and pipeline artifact to MLflow
# ============================================================
 
print("=" * 60)
print("PHASE 5 — TRAINING + MLFLOW LOGGING")
print("=" * 60)
 
experiment_path  = MODEL_CONFIG["mlflow"]["experiment_path"]
run_name         = MODEL_CONFIG["mlflow"]["run_name"]
primary_metric   = MODEL_CONFIG["evaluation"]["primary_metric"]
promo_threshold  = MODEL_CONFIG["evaluation"]["promotion_threshold"]
 
mlflow.set_experiment(experiment_path)
 
with mlflow.start_run(run_name=run_name) as run:
 
    run_id = run.info.run_id
    print(f"MLflow Run ID : {run_id}")
 
    # --- TRAIN ---
    print("\nTraining pipeline...")
    full_pipeline.fit(X_train, y_train)
 
    # --- PREDICT ---
    y_pred       = full_pipeline.predict(X_test)
    y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]
 
    # --- EVALUATE ---
    metrics = {
        "f1_score"  : round(f1_score(y_test, y_pred), 4),
        "roc_auc"   : round(roc_auc_score(y_test, y_pred_proba), 4),
        "precision" : round(precision_score(y_test, y_pred), 4),
        "recall"    : round(recall_score(y_test, y_pred), 4)
    }
 
    # --- LOG PARAMETERS ---
    mlflow.log_param("algorithm",       MODEL_CONFIG["model"]["algorithm"])
    mlflow.log_param("n_estimators",    MODEL_CONFIG["model"]["hyperparameters"]["n_estimators"])
    mlflow.log_param("max_depth",       MODEL_CONFIG["model"]["hyperparameters"]["max_depth"])
    mlflow.log_param("class_weight",    MODEL_CONFIG["model"]["class_weight"])
    mlflow.log_param("test_size",       test_size)
    mlflow.log_param("stratify",        stratify)
    mlflow.log_param("random_state",    random_state)
    mlflow.log_param("training_cohort", PIPELINE_CONFIG["data"]["filters"]["contract_cohorts"])
    mlflow.log_param("primary_metric",  primary_metric)
    mlflow.log_param("train_rows",      X_train.shape[0])
    mlflow.log_param("test_rows",       X_test.shape[0])
 
    # --- LOG METRICS ---
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
 
    # --- LOG PIPELINE ARTIFACT ---
    mlflow.sklearn.log_model(
        sk_model       = full_pipeline,
        artifact_path  = "pipeline",
        input_example  = X_train.head(5)
    )
 
    # --- PRINT RESULTS ---
    print("\n=== EVALUATION RESULTS ===")
    for k, v in metrics.items():
        marker = " ← PRIMARY" if k == primary_metric else ""
        print(f"  {k:12} : {v:.4f}{marker}")
 
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["No Churn", "Churn"]))
 
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- PRINT RESULTS ---
    print("\n=== EVALUATION RESULTS ===")
    for k, v in metrics.items():
        marker = " ← PRIMARY" if k == primary_metric else ""
        print(f"  {k:12} : {v:.4f}{marker}")

    # Explicitly bind variables to notebook global scope before context exit
    global_metrics = metrics
    global_run_id = run_id
 
# COMMAND ----------
# ============================================================
# CELL 9 — MODEL REGISTRY
# Promotes trained model to Unity Catalog Model Registry
# Registered as Version 1 — the baseline
# Only registers if primary metric meets promotion threshold
# ============================================================
 
print("=" * 60)
print("PHASE 6 — UNITY CATALOG MODEL REGISTRY")
print("=" * 60)

registered_model_name = MODEL_CONFIG["mlflow"]["registered_model_name"]
primary_score         = global_metrics[primary_metric] # Fixed scope reference

print(f"Primary metric    : {primary_metric} = {primary_score:.4f}")
print(f"Promotion threshold: {promo_threshold}")

if primary_score >= promo_threshold:
    model_uri = f"runs:/{global_run_id}/pipeline"
 
    registered_model = mlflow.register_model(
        model_uri  = model_uri,
        name       = registered_model_name
    )
 
    print(f"\nModel registered successfully!")
    print(f"  Name    : {registered_model.name}")
    print(f"  Version : {registered_model.version}")
    print(f"  URI     : {model_uri}")
    print(f"  Status  : Baseline Version 1 — registered to Unity Catalog")
 
else:
    print(f"\nPROMOTION BLOCKED")
    print(f"  {primary_metric} = {primary_score:.4f} is below threshold {promo_threshold}")
    print(f"  Model logged to MLflow but NOT registered to Model Registry")
    print(f"  Action: Tune hyperparameters and retrain")
 
# COMMAND ----------
# ============================================================
# CELL 10 — BASELINE SUMMARY
# Final audit of what was produced in this run
# Reference point for all future model versions
# ============================================================
 
print("=" * 60)
print("BASELINE TRAINING COMPLETE — RUN SUMMARY")
print("=" * 60)
 
print(f"""
  Run ID            : {run_id}
  Experiment        : {experiment_path}
  Training cohort   : {PIPELINE_CONFIG['data']['filters']['contract_cohorts']}
  Training rows     : {X_train.shape[0]}
  Test rows         : {X_test.shape[0]}
 
  METRICS:
    f1_score        : {metrics['f1_score']}
    roc_auc         : {metrics['roc_auc']}
    precision       : {metrics['precision']}
    recall          : {metrics['recall']}
 
  REGISTRY:
    Model name      : {registered_model_name}
    Version         : 1 (Baseline)
    Promoted        : {'YES' if primary_score >= promo_threshold else 'NO — below threshold'}
 
  NEXT STEP:
    Run 02_model_evaluation.py to validate baseline
    before introducing Month 2 drift data
""")