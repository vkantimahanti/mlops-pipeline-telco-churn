import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_score, recall_score,
    classification_report
)

from src.features import build_preprocessor


def train_model(df: pd.DataFrame,
                model_config: dict,
                pipeline_config: dict,
                feature_config: dict) -> None:
    """
    Full training run:
    split → preprocess → train → evaluate → log to MLflow

    Args:
        df:              Clean training dataframe
        model_config:    Loaded MODEL_CONFIG dict
        pipeline_config: Loaded PIPELINE_CONFIG dict
        feature_config:  Loaded FEATURE_CONFIG dict
    """
    # --- Config values ---
    target_col    = pipeline_config["data"]["target_column"]
    test_size     = model_config["training"]["test_size"]
    random_state  = model_config["model"]["random_state"]
    n_estimators  = model_config["model"]["hyperparameters"]["n_estimators"]
    max_depth     = model_config["model"]["hyperparameters"]["max_depth"]
    class_weight  = model_config["model"]["class_weight"]
    experiment    = model_config["mlflow"]["experiment_path"]
    run_name      = model_config["mlflow"]["run_name"]
    registered_name = model_config["mlflow"]["registered_model_name"]
    primary_metric= model_config["evaluation"]["primary_metric"]
    drop_cols     = pipeline_config["preprocessing"]["drop_columns"]


    # --- Prepare X and y ---
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col]

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y           # preserves class ratio
    )

    # --- Build preprocessor from config ---
    preprocessor = build_preprocessor(feature_config, pipeline_config)

    # --- Build full pipeline ---
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state
        ))
    ])

    # --- MLflow tracking ---
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):

        # Train
        full_pipeline.fit(X_train, y_train)
        y_pred      = full_pipeline.predict(X_test)
        y_pred_proba= full_pipeline.predict_proba(X_test)[:, 1]

        # Evaluate
        metrics = {
            "f1_score":  f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_pred_proba),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred)
        }

        # Log params
        mlflow.log_param("n_estimators",  n_estimators)
        mlflow.log_param("max_depth",     max_depth)
        mlflow.log_param("class_weight",  class_weight)
        mlflow.log_param("test_size",     test_size)
        mlflow.log_param("primary_metric",primary_metric)

        # Log metrics
        # Grab the catalog registry path from your config dictionary
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Generate structural data schema signature for deployment verification
        signature = infer_signature(X_train, y_pred)

        # Log full pipeline (preprocessor + model together)
        print(f"📦 Registering model to Unity Catalog: {registered_name}")
        mlflow.sklearn.log_model(
            sk_model=full_pipeline, 
            artifact_path="pipeline",
            registered_model_name=registered_name
        )

        # Print evaluation report
        print("\n=== EVALUATION RESULTS ===")
        for k, v in metrics.items():
            print(f"  {k:12}: {v:.4f}")
        print(f"\n{classification_report(y_test, y_pred)}")
        print(f"Primary metric ({primary_metric}): {metrics[primary_metric]:.4f}")

    print(f"\n✅ MLflow run logged and versioned successfully inside: {experiment}")