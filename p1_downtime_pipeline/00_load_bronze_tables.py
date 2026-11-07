# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Load Bronze tables (run once, by P1)
# MAGIC Reads every `.parquet` file in the raw volume (including sub-folders) and saves it **unchanged** as
# MAGIC `workspace.casino.bronze_<file_name>`. Bronze = raw copy. Nobody edits these tables.
# MAGIC
# MAGIC Safe to re-run: it overwrites. It also checks row counts against the official data manifest.

# COMMAND ----------

SCHEMA = "workspace.casino"
VOLUME = "/Volumes/workspace/casino/raw"

# Official row counts from the dashboard-royale data dictionary (for a quick sanity check)
EXPECTED_ROWS = {
    "prepared_daily_view": 56110, "property_day": 181, "player_period": 148893,
    "machine_day": 56110, "machines": 310, "game_math": 10, "placements": 314,
    "floor_locations": 378, "operations": 1226, "action_catalog": 13, "calendar_day": 181,
    "cost_rules": 9, "players": 113890, "sessions": 776297, "trips": 351729,
    "loyalty_ledger": 782773, "fnb_transactions": 227833, "zone_traffic_hourly": 30401,
    "events": 8, "shift_roster": 10317, "machine_hour": 1321908, "security_incidents": 3865,
}

def list_parquet(path):
    """Recursively list .parquet files in a volume folder."""
    out = []
    for f in dbutils.fs.ls(path):
        if f.isDir():
            out += list_parquet(f.path)
        elif f.name.endswith(".parquet"):
            out.append(f)
    return out

results = []
for f in list_parquet(VOLUME):
    name = f.name.replace(".parquet", "")
    table = f"{SCHEMA}.bronze_{name}"
    df = spark.read.parquet(f.path)
    (df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table))
    n = spark.table(table).count()
    exp = EXPECTED_ROWS.get(name)
    status = "OK" if exp in (None, n) else f"MISMATCH (expected {exp})"
    results.append((table, n, status))
    print(f"{table:55s} {n:>10,}  {status}")

# COMMAND ----------

# Which tables does P1 need that are still missing?
P1_NEEDS = ["machine_day", "prepared_daily_view", "calendar_day", "operations", "machines",
            "game_math", "player_period", "loyalty_ledger", "zone_traffic_hourly",
            "action_catalog", "sessions", "shift_roster"]
loaded = {t.split("bronze_", 1)[1] for t, _, _ in results}
missing = [t for t in P1_NEEDS if t not in loaded]
print("Missing for P1:", missing or "nothing")
if "shift_roster" in missing:
    print("-> Upload data/5_optional/shift_roster.parquet to the raw volume and re-run. "
          "The pipeline still runs without it, but loses the 'short-staffed shift' analysis.")

# COMMAND ----------

display(spark.table(f"{SCHEMA}.bronze_prepared_daily_view").limit(5))
