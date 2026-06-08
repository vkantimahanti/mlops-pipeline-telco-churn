import pandas as pd
from pyspark.sql.functions import col, when


def load_raw_to_delta(spark, pipeline_config: dict) -> None:
    """
    Reads CSV from Unity Catalog Volume → cleans → saves as Delta table.
    Run ONCE or when source data is refreshed.

    Args:
        spark: Active SparkSession
        pipeline_config: Loaded PIPELINE_CONFIG dict
    """
    volume_path = pipeline_config["data"]["baseline_csv"]
    raw_table   = pipeline_config["data"]["raw_table"]

    print(f"Reading from Volume: {volume_path}")

    spark_raw_df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(volume_path)
    )

    # Fix blank strings in TotalCharges → proper null
    spark_clean_df = spark_raw_df.withColumn(
        "TotalCharges",
        when(col("TotalCharges") == " ", None)
        .otherwise(col("TotalCharges").cast("double"))
    )

    spark_clean_df.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(raw_table)

    print(f"Raw Delta table created : {raw_table}")
    print(f"Total records           : {spark_clean_df.count()}")



def load_training_data(spark, pipeline_config: dict) -> pd.DataFrame:
    """
    Reads registered Delta table → applies distributed Spark filters → returns minimized Pandas df.
    Called every training run.
    """
    raw_table      = pipeline_config["data"]["raw_table"]
    filters        = pipeline_config["data"]["filters"]
    target_col     = pipeline_config["data"]["target_column"]
    positive_class = pipeline_config["data"]["positive_class"]

    print(f"Loading training data from: {raw_table}")

    # 1. Read the table as a distributed Spark DataFrame
    spark_df = spark.read.table(raw_table)

    # 2. Leverage distributed cluster compute for filtering
    spark_filtered = spark_df.filter(
        (col("Contract").isin(filters["contract_cohorts"])) & 
        (col("tenure") >= filters["min_tenure_months"])
    )

    # 3. Leverage distributed compute for target binary mapping
    spark_final = spark_filtered.withColumn(
        target_col,
        when(col(target_col) == positive_class, 1.0).otherwise(0.0)
    )

    # 4. Convert ONLY the optimized, smaller subset to Pandas for Scikit-Learn
    print("Converting optimized distributed cohort to Pandas...")
    df = spark_final.toPandas()

    print(f" Training data loaded successfully. Shape: {df.shape}")
    return df