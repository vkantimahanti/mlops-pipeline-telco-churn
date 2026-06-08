# Enterprise Telco Churn MLOps Pipeline
An end-to-end, configuration-driven, production-grade machine learning system built on Databricks. 

This repository implements **Infrastructure as Code (IaC)** via Databricks Asset Bundles (DABs), programmatic quality gatekeeping via Unity Catalog, defensive data sanitization against real-world data corruption, and horizontally scalable distributed batch scoring via PySpark.

---

## 🏛️ System Architecture Topology

The pipeline is orchestrated as a Directed Acyclic Graph (DAG) containing 5 sequential, isolated phases executed on automated Single-User compute clusters:


[00_Ingestion] ──► [01_Training] ──► [02_Gatekeeper] ──► [03_Stress_Simulation] ──► [04_Batch_Inference]

1. **`00_data_ingestion`**: Ingests raw historical billing snapshots from Unity Catalog Volumes, parses schemas, and archives them into optimized Delta Lake tables.
2. **`01_baseline_training_orchestrator`**: Extracts stable historical cohorts, logs metadata, parameters, and artifact binaries directly to MLflow tracking servers.
3. **`02_gatekeeper_metric_evaluator`**: Programmatically audits newly generated models against strict, decoupled business SLAs. Automatically promotes passing models to the `@champion` operational tag in Unity Catalog.
4. **`03_drift_and_stress_simulator`**: A chaos engineering block that exposes the active `@champion` to sudden covariate data drift (Month 2 cohort) and highly corrupted data strings (missing value tokens).
5. **`04_batch_inference_pipeline`**: A production scoring engine that automatically pulls the live `@champion`, runs parallelized inference across cluster worker nodes, and publishes predictive probability scores back to downstream business consumers.

---

## 🛡️ Key Architectural Enhancements & Edge Cases Handled

### 1. Zero-Crash Defensive Ingestion (Phase 3 Sandbox)
* **The Problem**: Real-world data feeds frequently drop fields, sending empty string literals (`' '`) instead of mathematical null spaces (`NaN`). Passing raw string blocks to a numeric `SimpleImputer` causes a catastrophic pipeline crash.
* **The Solution**: Implemented an upstream regex-driven sanitization layer (`^\s*$`) inside the unified feature preparation utilities to strip whitespace, coerce data types to floats, and safely convert empty strings to true `NaN` structures before downstream model evaluation.

### 2. High-Performance Distributed Batch Scoring (Phase 4 Scoring)
* **The Problem**: Standard single-node inference architectures utilize `.toPandas()` to pull entire datasets onto a single driver machine, creating immediate Out-Of-Memory (OOM) scaling bottlenecks when datasets scale to millions of rows.
* **The Solution**: Re-architected the batch scoring engine using `mlflow.pyfunc.spark_udf`. Inference processing is entirely parallelized across Spark worker nodes within the JVM, allowing the system to scale linearly with data volume growth.

### 3. Decoupled, Multi-Target Environment Isolation
* **The Problem**: Hardcoded infrastructure details create deployment vulnerabilities when moving code from development to production.
* **The Solution**: Environment settings, data targets, and hardware specifications are completely abstracted into a centralized `databricks.yml` deployment specification. Compute clusters dynamically scale from light testing resources in `dev` to robust worker counts in `prod` simply via deployment target flags.

---

## ⚙️ Configuration Layer Abstract Layout

Project execution thresholds and system data structures are controlled entirely through decoupled YAML manifests, enabling zero-code changes during post-deployment tuning:

* **`config/model_config.yaml`**: Governs model hyperparameter assignments, custom decision boundaries, and minimal performance promotion thresholds.
* **`config/pipeline_config.yaml`**: Coordinates source database locations, raw asset targets, and column dropping strategies.
* **`config/feature_config.yaml`**: Defines strict arrays for data types across numeric, low-cardinality, and high-cardinality data columns.

---

## 🚀 Deployment & Operations Playbook

This project is deployed completely as Infrastructure as Code via the Databricks CLI.

### Prerequisites
Ensure the Databricks CLI is authenticated and configured on your machine.

```bash
databricks auth login --host <your-databricks-instance-url>

databricks bundle validate

# Deploy to Staging/Development Environment (Default)
databricks bundle deploy --target dev

# Deploy to Production Workspace Environment
databricks bundle deploy --target prod

databricks bundle run --target dev telco_churn_end_to_end_job
