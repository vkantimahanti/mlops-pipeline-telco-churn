# Enterprise Telco Churn MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Databricks](https://img.shields.io/badge/Databricks-14.3_LTS-red?logo=databricks)
![MLflow](https://img.shields.io/badge/MLflow-2.9+-blue?logo=mlflow)
![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Governed-green)
![PySpark](https://img.shields.io/badge/PySpark-3.5+-orange?logo=apachespark)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn)
![DABs](https://img.shields.io/badge/Databricks_Asset_Bundles-IaC-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end, **configuration-driven, production-grade** machine learning system built on Databricks. This repository demonstrates ML Architect-level design patterns: zero-hardcoding config architecture, automated quality gatekeeping, covariate drift detection, chaos engineering stress tests, and Infrastructure as Code deployment via Databricks Asset Bundles.

---

## Table of Contents

- [Architecture](#architecture)
- [Pipeline DAG](#pipeline-dag)
- [Key Design Decisions](#key-design-decisions)
- [Configuration Layer](#configuration-layer)
- [Project Structure](#project-structure)
- [Performance Results](#performance-results)
- [Deployment Playbook](#deployment-playbook)
- [Local Development](#local-development)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UNITY CATALOG (Governance Layer)                 │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────────────┐    │
│  │   Volumes   │   │ Delta Tables │   │    Model Registry      │    │
│  │  (Raw CSV)  │   │  (Processed) │   │  @champion / @staging  │    │
│  └──────┬──────┘   └──────┬───────┘   └────────────┬───────────┘    │
└─────────│────────────────│────────────────────────│─────────────────┘
          │                │                        │
┌─────────▼────────────────▼────────────────────────▼───────────────────┐
│                    PIPELINE EXECUTION (DAG)                           │
│                                                                       │
│  [00_Ingest]──►[01_Train]──►[02_Gatekeeper]──►[03_Stress]──►[04_Score]│
│                                                                       │
│  config/ ──► src/ ──► notebooks/ ──► databricks.yml (IaC)             │
└───────────────────────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────────────────────┐
│                    MLFLOW TRACKING SERVER                            │
│    Parameters │ Metrics │ Artifacts │ Model Lineage │ Run History    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline DAG

The pipeline is a **5-stage Directed Acyclic Graph** where each stage depends on the success of the previous one. A failure at any stage halts the pipeline — preventing corrupt models from reaching production.

```
[00_data_ingestion]
      │
      │  Validates 3 source files (baseline + 2 stress test CSVs)
      │  Converts blank strings → true NaN before Delta registration
      │
      ▼
[01_baseline_training_orchestrator]
      │
      │  Filters stable cohorts (One year + Two year contracts)
      │  Builds config-driven sklearn Pipeline (impute→transform→scale→encode→model)
      │  Logs parameters, metrics, and full pipeline artifact to MLflow
      │
      ▼
[02_gatekeeper_metric_evaluator]
      │
      │  Pulls latest registered version from Unity Catalog
      │  Gates on primary metric (F1) vs promotion threshold
      │  Assigns @champion alias only if threshold is met
      │  Blocks deployment if model is underperforming
      │
      ▼
[03_drift_and_stress_simulator]
      │
      │  Test 1: Covariate drift — Month 2 cohort (all contracts)
      │  Test 2: Missing value resilience (blank strings, nulls)
      │  Test 3: Structural noise (unseen category values)
      │
      ▼
[04_batch_inference_pipeline]
      │
      │  Loads @champion via mlflow.pyfunc.spark_udf
      │  Parallelized scoring across Spark worker nodes
      │  Writes scored results back to Unity Catalog Delta table
      ▼
   [Done]
```

---

## Key Design Decisions

Every architectural decision in this project was made deliberately. Here is the reasoning behind each:

### 1. Three-File YAML Config Separation

**Decision:** Split configuration into `model_config.yaml`, `pipeline_config.yaml`, `feature_config.yaml` — not one monolithic config file.

**Why:**
```
model_config.yaml    → Data Scientist tunes this (hyperparams, metrics)
pipeline_config.yaml → Data Engineer manages this (paths, tables, filters)
feature_config.yaml  → Feature Engineer controls this (column definitions)

Single config file → all three roles editing the same file → merge conflicts
Three config files → independent ownership → parallel development
```

### 2. `config_loader.py` Lives in `src/`, Not in `config/`

**Decision:** YAML loading logic is a Python operation — it belongs in `src/`, keeping `config/` as pure data.

**Why:** The `config/` directory should be portable and framework-agnostic. Mixing Python operational code into `config/` violates Single Responsibility Principle and makes the config layer harder to audit.

### 3. OrdinalEncoder for Medium Cardinality, Not OneHotEncoder

**Decision:** Columns with 3–50 unique values use `OrdinalEncoder`, not `OneHotEncoder`.

**Why:**
```
Contract (3 values) via OHE    → adds 3 columns  (acceptable)
PaymentMethod (4 values) via OHE → adds 4 columns (acceptable)

At scale with 20 medium-cardinality features × 15 values avg:
OHE  → +300 columns → model overfits, inference slows, memory pressure
Ord  → +0 columns   → same information, stable dimensionality
```

### 4. `stratify=y` in Every Train/Test Split

**Decision:** Always stratify on the target variable.

**Why:** With a 73/27 class imbalance, random splits can produce training sets with 85% majority class and test sets with 60% — making evaluation metrics meaningless. Stratification guarantees identical class ratios in both splits.

### 5. Promotion Threshold Gate Before Registry

**Decision:** Models only reach the Unity Catalog Model Registry if F1 ≥ configured threshold. Failed models are logged to MLflow but never registered.

**Why:** MLflow logs everything by default. Without a gate, every experiment run — including bad ones — pollutes the registry. The gatekeeper ensures the registry only contains production-viable candidates.

### 6. `mlflow.pyfunc.spark_udf` for Batch Inference

**Decision:** Use Spark UDF for inference instead of `.toPandas()` + `model.predict()`.

**Why:**
```
.toPandas() approach:
  → Pulls ALL data to driver node RAM
  → 1M rows × 20 features = ~1.6GB on one machine
  → OOM error at scale

spark_udf approach:
  → Inference distributed across ALL worker nodes
  → Scales linearly with cluster size
  → 10M rows same cost as 1M rows with more workers
```

### 7. `00_data_ingestion` as a Separate Notebook

**Decision:** Data ingestion is isolated from training — not the first cell of the training notebook.

**Why:** Ingestion runs monthly (when source data refreshes). Training runs weekly (scheduled retraining). Coupling them forces unnecessary re-ingestion on every training run — slow, wasteful, and risky if the source file has changed unexpectedly.

### 8. `handle_unknown='ignore'` on All Encoders

**Decision:** All categorical encoders are configured to silently ignore unseen categories.

**Why:** Production data will always contain categories not seen during training — new contract types, payment methods, service offerings. Without `handle_unknown='ignore'`, the pipeline crashes on first encounter with new data. With it, unseen categories produce zeros (OHE) or -1 (Ordinal) — the model degrades gracefully instead of crashing.

---

## Configuration Layer

All pipeline behavior is controlled through three decoupled YAML manifests. Zero code changes are needed for tuning, environment migration, or dataset swaps.

```
config/
├── model_config.yaml      # algorithm, hyperparams, MLflow, metrics, thresholds
├── pipeline_config.yaml   # data sources, storage paths, filters, preprocessing rules
└── feature_config.yaml    # column lists, encoding strategies, cardinality classification
```

**Changing the model algorithm:**
```yaml
# model_config.yaml — one line change, no code touched
model:
  algorithm: "GradientBoostingClassifier"  # was RandomForestClassifier
```

**Adding a new feature:**
```yaml
# feature_config.yaml — add to correct cardinality group
categorical:
  low_cardinality:
    columns:
      - "gender"
      - "new_feature_name"   # added here, pipeline picks it up automatically
```

---

## Project Structure

```
mlops-pipeline-telco-churn/
│
├── config/                              # Pure YAML configuration (no Python)
│   ├── __init__.py
│   ├── model_config.yaml                # Model behavior & MLflow tracking
│   ├── pipeline_config.yaml             # Data sources, paths, filters
│   └── feature_config.yaml             # Feature definitions & encoding
│
├── notebooks/                           # Orchestration layer (thin callers)
│   ├── 00_data_ingestion.py             # Volume CSV → Delta table
│   ├── 01_baseline_training_orchestrator.py  # Train → MLflow → Registry
│   ├── 02_gatekeeper_metric_evaluator.py     # Validate → @champion alias
│   ├── 03_drift_and_stress_simulator.py      # Drift + chaos engineering
│   └── 04_batch_inference_pipeline.py        # Distributed scoring
│
├── src/                                 # Business logic (importable modules)
│   ├── __init__.py
│   ├── config_loader.py                 # YAML loader — exports MODEL/PIPELINE/FEATURE configs
│   ├── data_loader.py                   # load_raw_to_delta(), load_training_data()
│   ├── features.py                      # build_preprocessor() — config-driven sklearn pipeline
│   └── model_training.py               # train_model() — full MLflow run
│
├── tests/                               # Unit tests
│   ├── test_config_loader.py
│   └── test_features.py
│
├── .github/
│   └── workflows/
│       └── validate.yml                 # CI: bundle validate on every push
│
├── .gitignore
├── databricks.yml                       # IaC: DAB deployment spec (dev + prod)
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Performance Results

Baseline model results on held-out test set (20% stratified split):

| Metric | Score | Notes |
|---|---|---|
| F1 Score | 0.72 | Primary promotion gate metric |
| AUC-ROC | 0.84 | Discrimination across all thresholds |
| Precision | 0.69 | Of predicted churners, 69% actually churned |
| Recall | 0.76 | Of actual churners, 76% were caught |

**Class distribution:** 73.5% No Churn / 26.5% Churn → handled via `class_weight='balanced'`

**Drift simulation results (Month 2 cohort):**
- Month 1 Recall (stable cohorts): 0.76
- Month 2 Recall (all contracts): 0.61 — expected degradation, triggers retraining signal

**Stress test results:**
- Missing value test: PASS — SimpleImputer handled all null fields
- Structural noise test: PASS — `handle_unknown='ignore'` absorbed unseen categories

---

## Deployment Playbook

This project deploys entirely as Infrastructure as Code via the Databricks CLI.

### Prerequisites

```bash
# Install Databricks CLI
pip install databricks-cli

# Authenticate
databricks auth login --host <your-databricks-workspace-url>
```

### Validate

```bash
# Always validate before deploying — catches YAML syntax errors
databricks bundle validate
```

### Deploy

```bash
# Deploy to dev (default — schedule paused, smaller cluster)
databricks bundle deploy --target dev

# Deploy to prod (live schedule, larger cluster)
databricks bundle deploy --target prod
```

### Run

```bash
# Trigger full pipeline manually in dev
databricks bundle run --target dev telco_churn_end_to_end_job

# Run a single task for debugging
databricks bundle run --target dev telco_churn_end_to_end_job \
  --task 01_baseline_training
```

### Environment Differences

| Setting | dev | prod |
|---|---|---|
| Schedule | PAUSED (manual only) | UNPAUSED (every Monday 6am ET) |
| Node type | Standard_D4s_v5 | Standard_D8s_v5 |
| Workers | 2 | 4 |
| Job name prefix | [dev username] | [prod] |

---

## Local Development

```bash
# Clone repo
git clone https://github.com/vkantimahanti/mlops-pipeline-telco-churn.git
cd mlops-pipeline-telco-churn

# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/ -v
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Compute | Databricks Runtime 14.3 LTS ML |
| Storage | Unity Catalog Volumes + Delta Lake |
| ML Framework | scikit-learn 1.3+ |
| Experiment Tracking | MLflow 2.9+ |
| Model Registry | Unity Catalog Model Registry |
| Orchestration | Databricks Asset Bundles (DABs) |
| Distributed Scoring | PySpark + mlflow.pyfunc.spark_udf |
| CI/CD | GitHub Actions |
| Config | YAML (3-file separation pattern) |
| Language | Python 3.10+ |
