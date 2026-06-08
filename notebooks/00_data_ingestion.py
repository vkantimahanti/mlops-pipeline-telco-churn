# Databricks notebook source
# ============================================================
# NOTEBOOK  : 00_data_ingestion.py
# PURPOSE   : Load raw CSV from Unity Catalog Volume
#             → clean → register as Delta table
# TRIGGER   : Run ONCE before training pipeline
#             Re-run ONLY when source CSV is refreshed
# DEPENDS ON: CSV must exist in Volume path defined in
#             pipeline_config.yaml → data.volume_path
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
# CELL 2 — IMPORTS
# ============================================================

from src.config_loader import PIPELINE_CONFIG
from src.data_loader   import load_raw_to_delta

print("Imports successful")

# COMMAND ----------
# ============================================================
# CELL 3 — PRE-FLIGHT CHECK
# Verify Volume path exists before attempting to load
# ============================================================

volume_path = PIPELINE_CONFIG["data"]["baseline_csv"]
raw_table   = PIPELINE_CONFIG["data"]["raw_table"]

print("=" * 60)
print("PRE-FLIGHT CHECK")
print("=" * 60)
print(f"Source Volume path : {volume_path}")
print(f"Target Delta table : {raw_table}")

# Verify source file exists
try:
    files = dbutils.fs.ls(
        "/".join(volume_path.split("/")[:-1])  # parent directory
    )
    file_names = [f.name for f in files]
    source_file = volume_path.split("/")[-1]   # just the filename

    if source_file in file_names:
        print(f"\nSource file found  : {source_file}")
    else:
        print(f"\nSource file NOT found: {source_file}")
        print(f"Files in directory : {file_names}")
        raise FileNotFoundError(
            f"Expected file '{source_file}' not in Volume"
        )
except Exception as e:
    print(f"ERROR in pre-flight: {e}")
    raise

# COMMAND ----------
# ============================================================
# CELL 4 — LOAD VOLUME → DELTA TABLE
# This is the only cell that calls load_raw_to_delta()
# ============================================================

print("=" * 60)
print("INGESTION — Volume → Delta Table")
print("=" * 60)

load_raw_to_delta(spark, PIPELINE_CONFIG)

# COMMAND ----------
# ============================================================
# CELL 5 — VERIFY DELTA TABLE
# Confirm table was created and has expected data
# ============================================================

print("=" * 60)
print("VERIFICATION")
print("=" * 60)

verify_df = spark.read.table(raw_table)

print(f"Table name     : {raw_table}")
print(f"Total records  : {verify_df.count()}")
print(f"Total columns  : {len(verify_df.columns)}")
print(f"Columns        : {verify_df.columns}")

print("\nSample records:")
display(verify_df.limit(5))

print("\nSchema:")
verify_df.printSchema()

# COMMAND ----------
# ============================================================
# CELL 6 — INGESTION SUMMARY
# ============================================================

print("=" * 60)
print("INGESTION COMPLETE")
print("=" * 60)
print(f"""
  Source  : {volume_path}
  Target  : {raw_table}
  Records : {verify_df.count()}
  Status  : Delta table ready for training pipeline

  NEXT STEP:
    Run 01_baseline_training_orchestrator.py
""")