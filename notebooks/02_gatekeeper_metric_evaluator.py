# Databricks notebook source
# ============================================================
# NOTEBOOK  : 02_gatekeeper_metric_evaluator.py
# PURPOSE   : Automated Quality Governance & Alias Promotion
#             → Pulls latest registered version from Unity Catalog
#             → Validates performance against production targets
#             → Audits secondary diagnostic metrics
#             → Assigns '@champion' operational routing tag
# DEPENDS ON: 01_baseline_training_orchestrator.py
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
# CELL 1 — PATH ENVIRONMENT SETUP
# Resolve project root and register src/ on sys.path dynamically

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

print(f"Project root successfully mapped to: {project_root}")

# COMMAND ----------
# CELL 2 — GOVERNANCE ENGINE IMPORTS
# Load tracking clients and decoupled configuration maps
import mlflow
from mlflow.tracking import MlflowClient
from src.config_loader import MODEL_CONFIG

client = MlflowClient()

# Read the registered name from your config layer
model_name     = MODEL_CONFIG["mlflow"]["registered_model_name"]
target_metric  = MODEL_CONFIG["evaluation"]["primary_metric"]
pass_threshold = MODEL_CONFIG["evaluation"]["promotion_threshold"]

print(f" Governance Engine initialized for model: {model_name}")

# COMMAND ----------
# CELL 3 — FETCH LATEST VERSION UNDER TEST
# ============================================================
# CELL 3 — FETCH LATEST VERSION UNDER TEST (UNITY CATALOG SAFE)
# ============================================================
print("=" * 60)
print("FETCHING LATEST REGISTERED MODEL CANDIDATE")
print("=" * 60)

try:
    # Safely query model versions inside Unity Catalog using a search filter
    versions = client.search_model_versions(f"name='{model_name}'")
    
    if not versions:
        raise RuntimeError(f" Governance Error: No registered versions found for '{model_name}'.")
    
    # Sort by version number descending to ensure we grab the absolute newest candidate
    sorted_versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
    under_test_version = sorted_versions[0]
    
    v_num = under_test_version.version
    run_id = under_test_version.run_id

    print(f"Candidate Version Detected : Version {v_num}")
    print(f"Associated MLflow Run ID   : {run_id}")

except Exception as e:
    print(f" Error connecting to Unity Catalog Registry: {e}")
    print("Please make sure Notebook 1 completed successfully and registered the model assets.")
    raise e

# COMMAND ----------
# CELL 4 — METRIC VERIFICATION GATE & METRIC REPORT
print("=" * 60)
print("RUNNING GOVERNANCE PERFORMANCE AUDIT")
print("=" * 60)

# Pull configuration values safely
secondary_metrics  = MODEL_CONFIG["evaluation"].get("secondary_metrics", [])
decision_threshold = MODEL_CONFIG["evaluation"].get("decision_threshold", 0.5)

# Extract out the logged training run data directly via the tracking API
run_info = client.get_run(run_id)
logged_metrics = run_info.data.metrics

candidate_primary_score = logged_metrics.get(target_metric)
if candidate_primary_score is None:
    raise ValueError(f" Governance Error: Target metric '{target_metric}' was not found logged in run '{run_id}'.")

# 1. Print Holistic Diagnostic Dashboard
print(f"Configured Decision Threshold : {decision_threshold}")
print("\n Secondary Metrics Diagnostic Scan:")
for metric in secondary_metrics:
    val = logged_metrics.get(metric, 0.0)
    print(f"  • {metric:12} : {val:.4f}")

print("\n" + "-"*60)
print(f"Gating Check: Is {target_metric} ({candidate_primary_score:.4f}) >= Threshold ({pass_threshold:.4f})?")
print("-"*60)

# 2. Final Deployment Decision Gate
if candidate_primary_score >= pass_threshold:
    print(f" PASS: Version {v_num} clears the primary '{target_metric}' quality gate.")
    print(f"Promoting Version {v_num} to official production '@champion' status...")
    
    # Programmatically assign the alias tag directly within Unity Catalog
    client.set_registered_model_alias(
        name=model_name, 
        alias="champion", 
        version=str(v_num)
    )
    
    print(f"\n SUCCESS: Model path '{model_name}@champion' is now live!")
    print(f"Future inference steps will automatically route straight to Version {v_num}.")
else:
    print(f" FAIL: Version {v_num} primary score is insufficient.")
    print(f"Operational promotion blocked. Action required: Re-tune hyperparameters in config files.")