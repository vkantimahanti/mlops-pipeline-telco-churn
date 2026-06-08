# Databricks notebook source
# ============================================================
# NOTEBOOK  : 04_batch_inference_pipeline.py
# PURPOSE   : Automated Production Score Generation
#             → Load live @champion model from Unity Catalog
#             → Ingest current week's active customer dataset
#             → Run defensive feature cleaning using prepare_features
#             → Generate batch churn risk predictions
#             → Overwrite/Append results to business consumer table
# DEPENDS ON: Databricks Workflow Schedule (Weekly/Monthly)
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
# CELL 1 — ENVIRONMENT SETUP

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
project_root = "/Workspace" + os.path.dirname(os.path.dirname(notebook_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# COMMAND ----------
# CELL 2 — CONFIGURATION & CHAMPION LOADING
import mlflow.sklearn
from src.config_loader import MODEL_CONFIG, PIPELINE_CONFIG, FEATURE_CONFIG

# Ingest active decoupled metadata configurations
model_name = MODEL_CONFIG["mlflow"]["registered_model_name"]
target_col = PIPELINE_CONFIG["data"]["target_column"]
pos_class  = PIPELINE_CONFIG["data"]["positive_class"]
drop_cols  = PIPELINE_CONFIG["preprocessing"]["drop_columns"]
raw_table  = PIPELINE_CONFIG["data"]["raw_table"]

# Target the live routing tag inside Unity Catalog
champion_uri = f"models:/{model_name}@champion"
print(f" Establishing connection to Unity Catalog...")
print(f" Fetching live production artifact from: {champion_uri}")

champion_model = mlflow.sklearn.load_model(champion_uri)
print(" Production @champion model loaded successfully into pipeline RAM.")

# COMMAND ----------
# CELL 3 — HARDENED FEATURE PREPARATION UTILITY
def prepare_production_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Production-grade feature isolation utility. 
    Cleans raw whitespace and isolates features cleanly without target variables.
    """
    df = df.copy()

    #  DEFENSIVE REPAIR: Convert empty space strings (' ') to true NaNs
    df = df.replace(r'^\s*$', np.nan, regex=True)

    #  FORCE TYPE CASTING: Ensure known numeric features are strictly floats
    num_cols = FEATURE_CONFIG.get("features", {}).get("numeric", {}).get("columns", ['tenure', 'MonthlyCharges', 'TotalCharges'])
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Isolate features (Production data will NOT contain the 'Churn' target column)
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    return X

# COMMAND ----------
# COMMAND ----------
# CELL 4 — DISTRIBUTED INGESTION (NO MORE PANAS CONVERSION)
print("=" * 60)
print("OPTIMIZED PHASE 1 — DISTRIBUTED PRODUCTION DATA INGESTION")
print("=" * 60)

print(f"Reading active production snapshots from Delta Lake: {raw_table}")

# Keep the data as a distributed Spark DataFrame. Do NOT use .toPandas()
df_production_spark = spark.read.table(raw_table)
print(f"Initial distributed DataFrame successfully targeted.")

# COMMAND ----------
# CELL 5 — HIGH-PERFORMANCE MLFLOW PYSPARK INFERENCE
print("=" * 60)
print("OPTIMIZED PHASE 2 — DISTRIBUTED SCORING VIA MLFLOW UDF")
print("=" * 60)

import mlflow

# 1. Register your champion model as a distributed Spark UDF
# MLflow automatically handles feature alignment and serialization behind the scenes
print("Registering model into Spark Engine execution context...")
distributed_predict_udf = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri=champion_uri, 
    result_type="double"
)

# 2. Extract the exact feature list the model expects (excluding IDs and Target)
feature_columns = [
    col for col in df_production_spark.columns 
    if col not in ([target_col] + drop_cols)
]

# 3. Execute distributed inference across the cluster nodes
print("Executing parallelized inference across cluster worker nodes...")
df_scored_spark = df_production_spark.withColumn(
    "predicted_churn_risk", 
    distributed_predict_udf(*feature_columns)
)

# COMMAND ----------
# CELL 6 — DIRECT DELTA LAKE WRITING
# OPTIMIZED PHASE 3 — BULK DELTA ARCHIVING
print("=" * 60)
print("OPTIMIZED PHASE 3 — BULK DELTA ARCHIVING")
print("=" * 60)

output_table_name = f"{raw_table}_scored_predictions"

# THE FIX: Clean up old legacy schema definitions to prevent metadata type conflicts
print(f"Dropping stale legacy table metadata for: {output_table_name}")
spark.sql(f"DROP TABLE IF EXISTS {output_table_name}")

print(f"Writing parallelized predictions directly to: {output_table_name}")

# Write the data directly back to Unity Catalog using Spark's distributed writer
(
    df_scored_spark.write
    .mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(output_table_name)
)

print(f"\n✅ OPTIMIZATION COMPLETE: Big Data scaling bottleneck eliminated.")
print("The inference pipeline can now process millions of customers seamlessly without OOM risks.")