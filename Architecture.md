# Architecture Decision Record (ADR)

> **Purpose:** Documents the WHY behind every major design decision in this MLOps pipeline.
> This is what separates an ML Architect from an ML Engineer — not just what was built, but why.

---

## Table of Contents

1. [System Layers Overview](#system-layers-overview)
2. [Data Layer Decisions](#data-layer-decisions)
3. [Feature Engineering Decisions](#feature-engineering-decisions)
4. [Model Training Decisions](#model-training-decisions)
5. [Deployment & Infrastructure Decisions](#deployment--infrastructure-decisions)
6. [Config Architecture Decisions](#config-architecture-decisions)
7. [Observability Decisions](#observability-decisions)
8. [What Was Deliberately Left Out](#what-was-deliberately-left-out)

---

## System Layers Overview

```
┌─────────────────────────────────────────────────────┐
│  LAYER 1: Infrastructure (databricks.yml)            │
│  IaC deployment, environment isolation, scheduling   │
├─────────────────────────────────────────────────────┤
│  LAYER 2: Orchestration (notebooks/)                 │
│  Thin callers — no business logic, only sequencing   │
├─────────────────────────────────────────────────────┤
│  LAYER 3: Business Logic (src/)                      │
│  All ML logic — importable, testable, reusable       │
├─────────────────────────────────────────────────────┤
│  LAYER 4: Configuration (config/)                    │
│  Pure YAML — no Python, no logic, only values        │
├─────────────────────────────────────────────────────┤
│  LAYER 5: Governance (Unity Catalog + MLflow)        │
│  Model registry, data lineage, access control        │
└─────────────────────────────────────────────────────┘
```

**Key principle:** Each layer has one job. Notebooks never contain ML logic. `src/` never reads from files directly. `config/` never imports Python.

---

## Data Layer Decisions

### ADR-001: Separate Ingestion Notebook from Training Notebook

**Status:** Accepted

**Context:** Initial design had ingestion as the first cell of the training notebook.

**Decision:** `00_data_ingestion.py` is a standalone notebook, never called by `01_baseline_training_orchestrator.py`.

**Consequences:**
```
+ Ingestion runs on its own schedule (monthly data refresh)
+ Training runs on its own schedule (weekly retraining)
+ Failure in source system doesn't block model retraining
+ Delta table acts as stable contract between both stages
- Requires two separate DAB task definitions
- Developer must remember to run ingestion before first training
```

**Alternatives rejected:**
- Ingestion as Cell 1 of training notebook → couples two different schedules
- Ingestion as a Python function called from training → same coupling problem

---

### ADR-002: Delta Lake as Intermediate Layer Between Volume and Model

**Status:** Accepted

**Context:** Could read directly from Volume CSV on every training run.

**Decision:** Always write to Delta table first, then read Delta table for training.

**Consequences:**
```
+ Delta provides ACID transactions — no partial reads mid-write
+ Schema enforcement catches upstream data type changes
+ Time travel allows rollback to previous dataset version
+ Multiple notebooks can read same table simultaneously
+ Query performance significantly faster than CSV at scale
- Extra ingestion step required
- Storage cost for Delta table (minimal)
```

---

### ADR-003: Blank String → NaN Conversion at Ingestion Time

**Status:** Accepted

**Context:** Spark's CSV reader treats blank values as `" "` (space string), not `null`.

**Decision:** Explicit `when(trim(col) == "", None)` applied to every column during ingestion.

**Consequences:**
```
+ SimpleImputer receives proper NaN — no crash
+ Consistent null semantics across the pipeline
+ Stress test files also cleaned via same utility function
- Slight overhead during ingestion for all-column scan
```

**Why not fix at training time?**
Fixing at training time means the bug exists in your Delta table. Every consumer of that table inherits the problem. Fix it once at source.

---

## Feature Engineering Decisions

### ADR-004: Three-Pipeline ColumnTransformer Design

**Status:** Accepted

**Decision:** Separate sklearn sub-pipelines for skewed numeric, normal numeric, OHE categorical, and Ordinal categorical columns.

```python
ColumnTransformer([
    ("skewed_num", skewed_pipeline,  skewed_cols),    # impute→log→scale
    ("num",        numeric_pipeline, normal_cols),     # impute→scale
    ("ohe",        ohe_pipeline,     low_card_cols),   # impute→OHE
    ("ord",        ord_pipeline,     med_card_cols),   # impute→Ordinal
])
```

**Why not a single pipeline for all numeric?**
```
TotalCharges skewness = 0.85 → needs log transform
tenure       skewness = 0.02 → does NOT need log transform

Applying log to all numeric:
  log(0)    → -inf  (breaks model)
  log(neg)  → NaN   (breaks model)

Separate pipelines → correct transform applied only where needed
```

---

### ADR-005: OrdinalEncoder for Medium Cardinality (3–50 values)

**Status:** Accepted

**Decision:** Columns with 3–50 unique values use `OrdinalEncoder`, not `OneHotEncoder`.

**Threshold reasoning:**
```
Low cardinality (≤10 values)  → OHE: adds ≤10 columns (safe)
Medium cardinality (10–50)    → Ordinal: adds 0 columns (efficient)
High cardinality (>50)        → Drop or target encode
```

**Risk acknowledged:** OrdinalEncoder implies numeric ordering where none exists. Mitigated by: tree-based models (RandomForest, XGBoost) don't interpret ordinal values as ordered — they split on them as thresholds.

---

### ADR-006: `remainder='drop'` in ColumnTransformer

**Status:** Accepted

**Decision:** Any column not explicitly listed in feature groups is silently dropped.

**Why:** Identifier columns (`customerID`) must never reach the model. Explicit drop lists can be forgotten. `remainder='drop'` as default ensures only intentionally included columns survive preprocessing — defense-in-depth against feature leakage.

---

## Model Training Decisions

### ADR-007: RandomForest as Baseline (Not XGBoost)

**Status:** Accepted

**Context:** XGBoost typically outperforms RandomForest on tabular data.

**Decision:** RandomForest as baseline model version 1.

**Reasoning:**
```
Baseline purpose: establish a performance floor, not ceiling
RandomForest:
  → No hyperparameter sensitivity (trees are stable)
  → No learning rate to tune
  → Feature importance natively available
  → Interpretable to non-technical stakeholders

XGBoost reserved for: Version 2 after baseline is validated
```

---

### ADR-008: `class_weight='balanced'` Over SMOTE for Initial Pipeline

**Status:** Accepted

**Context:** Dataset has 73/27 class imbalance.

**Decision:** Use `class_weight='balanced'` parameter, not SMOTE oversampling.

**Reasoning:**
```
SMOTE:
  → Creates synthetic minority samples
  → Must be applied ONLY inside cross-validation folds
  → Easy to introduce data leakage if applied before split
  → Adds imbalanced-learn dependency and pipeline complexity

class_weight='balanced':
  → Native sklearn parameter — zero extra dependencies
  → No data creation — mathematically equivalent weighting
  → Cannot leak — applied during loss computation, not data prep
  → Simpler → easier to audit → safer for production baseline

SMOTE reserved for: Future experiment if balanced weights insufficient
```

---

### ADR-009: Full sklearn Pipeline Logged to MLflow (Not Just Model)

**Status:** Accepted

**Decision:** `mlflow.sklearn.log_model(full_pipeline)` logs the complete Pipeline object including preprocessor — not just the RandomForest estimator.

**Why:**
```
Logging model only:
  → Inference requires separate preprocessing code
  → Preprocessing code can drift from training code
  → Two artifacts to version and deploy

Logging full pipeline:
  → Single artifact: preprocessing + model
  → Inference = load artifact + predict (no extra code)
  → Guaranteed identical preprocessing at training and inference
  → mlflow.pyfunc.spark_udf works directly on raw features
```

---

### ADR-010: F1 Score as Primary Promotion Gate Metric

**Status:** Accepted

**Context:** Could use accuracy, AUC-ROC, precision, or recall.

**Decision:** F1 score as primary metric for gatekeeper promotion threshold.

**Reasoning:**
```
Accuracy:    Misleading at 73/27 imbalance
             (73% by predicting "No" always)

Precision:   Optimizes for "don't cry wolf"
             (minimize false alarms)
             Wrong goal for churn — we want to CATCH churners

Recall:      Maximizes catching churners but ignores false positives
             Sending retention offers to 100% of customers is expensive

AUC-ROC:     Good for ranking, less intuitive for business threshold

F1:          Harmonic mean of precision and recall
             Penalizes models that sacrifice one for the other
             Best single metric for imbalanced classification
             where both false positives and false negatives have cost
```

---

## Deployment & Infrastructure Decisions

### ADR-011: Databricks Asset Bundles Over REST API Deployment

**Status:** Accepted

**Decision:** Deploy via `databricks.yml` DAB spec, not manual REST API calls or Terraform.

**Reasoning:**
```
REST API:    Imperative — run scripts to create jobs
             Not version controlled
             Not reproducible

Terraform:   Excellent for infrastructure, complex for ML pipelines
             Requires separate state management
             Databricks provider lags behind DAB features

DABs:        Declarative — describe what you want
             Native to Databricks — first-class support
             Version controlled alongside code
             Environment variables built in
             CI/CD integration via CLI
```

---

### ADR-012: `data_security_mode: SINGLE_USER`

**Status:** Accepted

**Decision:** All job clusters use `SINGLE_USER` access mode.

**Why:**
```
NO_ISOLATION:    Cannot access Unity Catalog Volumes
                 Cannot use Unity Catalog Model Registry
                 Not supported for production ML workloads

SHARED:          Multi-tenant — performance unpredictable
                 Not suitable for scheduled production jobs

SINGLE_USER:     Full Unity Catalog access (Volumes, Registry, Tables)
                 Dedicated compute — predictable performance
                 Required for mlflow.pyfunc.spark_udf
```

---

### ADR-013: Task-Level Libraries, Not Cluster-Level

**Status:** Accepted

**Decision:** Python libraries declared per-task in DAB spec, not in cluster definition.

**Why:**
```
Cluster-level libraries:
  → Installed once, shared by all tasks
  → Version conflicts if tasks need different library versions
  → Cannot easily update one task's dependencies independently

Task-level libraries:
  → Installed fresh for each task run
  → Each task declares exactly what it needs
  → Future tasks can use different versions
  → Easier to debug dependency issues per task
```

---

## Config Architecture Decisions

### ADR-014: `config_loader.py` in `src/`, Not in `config/`

**Status:** Accepted

**Decision:** YAML loading is a Python operation — it lives in `src/config_loader.py`. The `config/` directory contains only YAML files and `__init__.py`.

**Why:**
```
config/ purpose:  Store declarative values
                  Should be readable by non-developers
                  Should be portable to any framework

Python in config/:
  → Breaks the contract that config/ is pure data
  → Makes config/ depend on Python environment
  → Cannot be validated independently of Python

Python in src/:
  → Business logic belongs with business logic
  → Testable independently
  → Clear separation of concerns
```

---

### ADR-015: Three Separate YAML Files vs One Monolithic Config

**Status:** Accepted

**Decision:** Three files: `model_config.yaml`, `pipeline_config.yaml`, `feature_config.yaml`.

**Ownership model:**
```
Role                 Owns                      Never edits
───────────────────────────────────────────────────────────
Data Scientist   →   model_config.yaml     →   pipeline_config
Data Engineer    →   pipeline_config.yaml  →   model_config
Feature Engineer →   feature_config.yaml   →   pipeline_config
```

**Risk:** More files to manage. Mitigated by: each file is small, focused, and self-documenting. A monolithic config file grows unmanageable as the pipeline scales.

---

## Observability Decisions

### ADR-016: MLflow Parameters Include `training_cohort`

**Status:** Accepted

**Decision:** Log `training_cohort` (the filter applied to training data) as an MLflow parameter alongside hyperparameters.

**Why:**
```
Standard MLflow logging:
  n_estimators=100, max_depth=4, test_size=0.2

This pipeline adds:
  training_cohort=["One year", "Two year"]

Purpose:
  When comparing Month 1 vs Month 2 models in MLflow UI,
  you can immediately see which data cohort each model
  was trained on — without reading source code.

  This is the difference between:
    "Why did recall drop in this run?"
    "Because this run trained on Month-to-month contracts too"
```

---

## What Was Deliberately Left Out

Understanding what was NOT built is as important as what was built:

| Excluded | Why | When to Add |
|---|---|---|
| LangGraph / LangChain | Adds complexity not needed for batch ML pipeline | When adding LLM-based feature generation or RAG |
| Feature Store | Overkill for single-project portfolio | When features are shared across multiple models |
| Model Serving (REST endpoint) | Batch inference sufficient for churn use case | When real-time churn scoring is required |
| Hyperparameter tuning (Hyperopt) | Baseline first, optimize second | After baseline metrics establish floor |
| SMOTE | `class_weight='balanced'` is simpler and safer | If F1 remains below threshold after multiple runs |
| dbt | No complex SQL transformation layer needed | When adding a governed semantic layer |
| Great Expectations | Config-based validation sufficient at this scale | When data contracts need formal enforcement |
