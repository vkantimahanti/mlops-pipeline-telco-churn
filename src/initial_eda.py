# Databricks notebook source
dbutils.fs.ls("/Volumes/databricks_telco_customer_dataset/v01/telco/")

# COMMAND ----------

# Databricks notebook source
from pyspark.sql.functions import col, when

# Ingest directly from your Databricks Volume path
volume_path = "/Volumes/databricks_telco_customer_dataset/v01/telco/telco-customer-churn.csv"

# Load with explicit header and schema inference
spark_raw_df = (spark.read.format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load(volume_path))

# Clean the data types using Spark functions instead of Pandas
# This resolves the type mismatch error shown in your screenshot
spark_clean_df = spark_raw_df.withColumn(
    "TotalCharges", 
    when(col("TotalCharges") == " ", 0.0).otherwise(col("TotalCharges").cast("double"))
)

# Save as a managed Delta Table for your pipeline
spark_clean_df.write.format("delta").mode("overwrite").saveAsTable("workspace.default.telco_churn_raw")

print(f" Data processed from Volume to Delta Table. Total records: {spark_clean_df.count()}")

# COMMAND ----------

# Databricks notebook source
import pandas as pd
import numpy as np

# 1. Load data from the Delta table we created from your Volume
raw_df = spark.read.table("workspace.default.telco_churn_raw")
df_eda = raw_df.toPandas()

print(f" DATASET PROFILE LOADED. Shape: {df_eda.shape} (Rows, Columns)\n")

print("==================================================================")
print(" PHASE 1: STRUCTURAL INTEGRITY & DATA TYPE VERIFICATION")
print("==================================================================")
# Why: Validates if Spark/Pandas interpreted types correctly. 
# String object types need encoding; numerical types need scaling.
print(df_eda.info())

print("\n==================================================================")
print(" PHASE 2: COMPLETENESS PROFILE (Null & Missing Value Analysis)")
print("==================================================================")
# Why: In production, missing values cause models to crash. 
# This tells us exactly where we need to deploy SimpleImputers.
null_counts = df_eda.isnull().sum()
null_percentages = (df_eda.isnull().sum() / len(df_eda)) * 100
null_profile = pd.DataFrame({'Total Nulls': null_counts, 'Percentage (%)': null_percentages})
print(null_profile[null_profile['Total Nulls'] > 0].sort_values(by='Total Nulls', ascending=False))

print("\n==================================================================")
print(" PHASE 3: CARDINALITY PROFILE (Categorical Complexity)")
print("==================================================================")
# Why: High cardinality (too many unique strings) causes dimensionality explosions 
# during One-Hot Encoding. Low cardinality means clean grouping.
categorical_cols = df_eda.select_dtypes(include=['object', 'string']).columns
for col in categorical_cols:
    unique_count = df_eda[col].nunique()
    print(f"Column '{col}' has {unique_count} unique values.")

print("\n==================================================================")
print(" PHASE 4: SUPERVISED LEARNING TARGET CHECK (Class Balance Analysis)")
print("==================================================================")
# Why: Determines if Target Imbalance exists. If one class dominates (e.g., 90% No Churn),
# standard accuracy metrics are useless, and you must use balanced class weights.
if 'Churn' in df_eda.columns:
    target_counts = df_eda['Churn'].value_counts()
    target_pct = df_eda['Churn'].value_counts(normalize=True) * 100
    print("Class Distribution Counts:")
    print(target_counts)
    print("\nClass Distribution Percentages:")
    print(target_pct)

print("\n==================================================================")
print(" Skweness")
print("==================================================================")
skewness = df_eda[numeric_cols].skew()
print(skewness)

print("\n==================================================================")
print(" PHASE 5: UNSUPERVISED/SUPERVISED METRIC SPREAD (Continuous Stats)")
print("==================================================================")
# Why: Shows skewness and scale. If 'TotalCharges' is in the thousands and 'tenure' 
# is under 100, unsupervised distance algorithms (like K-Means) will be warped 
# unless we use a StandardScaler.
numeric_cols = df_eda.select_dtypes(include=[np.number]).columns
print(df_eda[numeric_cols].describe().T)

print("\n==================================================================")
print(" PHASE 6: CORRELATION & LEAKAGE DETECTIVE (Cross-Tabulation)")
print("==================================================================")
# Why: Identifies structural leakage. If Churn is perfectly predictable by 
# a single category feature, your model will cheat during training and fail in production.
important_features = ['Contract', 'InternetService', 'PaymentMethod']
for feature in important_features:
    print(f"\nCross-tabulation: {feature} vs Churn")
    print(pd.crosstab(df_eda[feature], df_eda['Churn'], margins=True, normalize='index') * 100)