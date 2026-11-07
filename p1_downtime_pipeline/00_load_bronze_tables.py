# Databricks notebook source
# Loads every parquet file in the raw volume into a shared bronze_ table.
volume_path = "/Volumes/workspace/casino/raw"

for f in dbutils.fs.ls(volume_path):
    if f.name.endswith(".parquet"):
        table_name = "bronze_" + f.name.replace(".parquet", "")
        df = spark.read.parquet(f.path)
        df.write.mode("overwrite").saveAsTable(f"workspace.casino.{table_name}")
        print(f"Created {table_name}: {df.count()} rows")
