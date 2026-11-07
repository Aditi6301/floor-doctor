# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # P3 · 00 Explore tables
# MAGIC Run this first. It shows the columns of every table Person 3 needs.
# MAGIC **Copy the printed output and send it to Claude** so the next notebooks use the exact column names.

# COMMAND ----------

tables = [
    "bronze_property_day",
    "bronze_calendar_day",
    "bronze_events",
    "bronze_zone_traffic_hourly",
    "bronze_prepared_daily_view",
    "bronze_player_period",
    "bronze_players",
    "bronze_action_catalog",
]

for t in tables:
    full = f"workspace.casino.{t}"
    try:
        df = spark.table(full)
        print(f"\n=== {t}  ({df.count():,} rows) ===")
        for name, dtype in df.dtypes:
            print(f"  {name}: {dtype}")
    except Exception as e:
        print(f"\n=== {t}: NOT FOUND ({e.__class__.__name__}) ===")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Which real-source files are in the volume?
# MAGIC If none show up, upload `data/4_context/real_sources/*.parquet` to `workspace.casino.raw`.

# COMMAND ----------

for f in dbutils.fs.ls("/Volumes/workspace/casino/raw"):
    print(f.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick look at property_day

# COMMAND ----------

display(spark.table("workspace.casino.bronze_property_day").limit(10))