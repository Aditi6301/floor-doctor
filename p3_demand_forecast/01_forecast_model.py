# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # P3 · 01 Forecast model (required)
# MAGIC **Goal:** predict daily floor coin-in.
# MAGIC - Train: Jan 1 to Apr 30, 2025 · Test: May 1 to Jun 30, 2025 (time-based holdout)
# MAGIC - Baseline: constant training-period daily mean
# MAGIC - Model 1: average by weekday · Model 2: weekday + holiday + known events
# MAGIC - Metric: MAE in dollars · Every run logged in MLflow
# MAGIC
# MAGIC **No leakage:** only information known before each test day is used. Test data is never used to build features.

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import mean_absolute_error

pdf = spark.table("workspace.casino.bronze_property_day").toPandas()
print(list(pdf.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pick the date and coin-in columns
# MAGIC The code guesses them. **Check the printout.** If a guess is wrong, type the right name in the override lines.

# COMMAND ----------

def find_col(cols, must_have, prefer=None):
    hits = [c for c in cols if all(k in c.lower() for k in must_have)]
    if prefer:
        pref = [c for c in hits if prefer in c.lower()]
        if pref:
            return pref[0]
    return hits[0] if hits else None

DATE_COL = find_col(pdf.columns, ["gaming_date"]) or find_col(pdf.columns, ["date"])
COIN_COL = find_col(pdf.columns, ["coin_in"], prefer="usd")

# Overrides (fill in if the guesses are wrong):
# DATE_COL = "gaming_date"
# COIN_COL = "coin_in_usd"

print("Date column:   ", DATE_COL)
print("Coin-in column:", COIN_COL)

# COMMAND ----------

df = pdf[[DATE_COL, COIN_COL]].copy()
df.columns = ["gaming_date", "coin_in_usd"]
df["gaming_date"] = pd.to_datetime(df["gaming_date"])
df["coin_in_usd"] = pd.to_numeric(df["coin_in_usd"], errors="coerce")
if "cents" in COIN_COL.lower():
    df["coin_in_usd"] = df["coin_in_usd"] / 100   # convert cents to dollars
df = df.sort_values("gaming_date").reset_index(drop=True)
df["weekday"] = df["gaming_date"].dt.day_name()

print(df["gaming_date"].min().date(), "to", df["gaming_date"].max().date(), "|", len(df), "days")
print(df["coin_in_usd"].describe().round(0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Holiday and known-event flags
# MAGIC Holidays come from `calendar_day`. Events come from `events`, but only events whose dates are known ahead
# MAGIC (conventions, sports weekends, festivals, scheduled promotions), so using them is not leakage.

# COMMAND ----------

cal = spark.table("workspace.casino.bronze_calendar_day").toPandas()
cal_date = find_col(cal.columns, ["gaming_date"]) or find_col(cal.columns, ["date"])
hol_col = find_col(cal.columns, ["holiday"])
print("calendar columns:", list(cal.columns))
print("holiday column guess:", hol_col)

df["is_holiday"] = 0
if hol_col:
    cal[cal_date] = pd.to_datetime(cal[cal_date])
    hol = cal[[cal_date, hol_col]].rename(columns={cal_date: "gaming_date", hol_col: "hol"})
    # True if the holiday column is True/1 or holds a holiday name
    txt = hol["hol"].astype(str).str.strip().str.lower()
    hol["hol"] = ~txt.isin(["false", "0", "none", "nan", "", "null"])
    df = df.merge(hol, on="gaming_date", how="left")
    df["is_holiday"] = df["hol"].fillna(False).astype(int)
    df = df.drop(columns="hol")

# COMMAND ----------

ev = spark.table("workspace.casino.bronze_events").toPandas()
print("events columns:", list(ev.columns))
display(ev)

# COMMAND ----------

start_col = find_col(ev.columns, ["start"])
end_col = find_col(ev.columns, ["end"])
name_col = find_col(ev.columns, ["name"]) or find_col(ev.columns, ["event"])
print("event start/end/name guesses:", start_col, end_col, name_col)

df["is_event"] = 0
df["event_name"] = None
if start_col and end_col:
    for _, e in ev.iterrows():
        s = pd.to_datetime(str(e[start_col])[:10])
        t = pd.to_datetime(str(e[end_col])[:10])
        mask = (df["gaming_date"] >= s) & (df["gaming_date"] <= t)
        df.loc[mask, "is_event"] = 1
        df.loc[mask, "event_name"] = str(e[name_col]) if name_col else "event"
print(df[["is_holiday", "is_event"]].sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Time-based split

# COMMAND ----------

train = df[df["gaming_date"] < "2025-05-01"].copy()
test = df[df["gaming_date"] >= "2025-05-01"].copy()
print(len(train), "train days |", len(test), "test days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Baseline and models, logged in MLflow

# COMMAND ----------

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/floor_doctor_forecast")

def log_run(name, method, preds):
    mae = mean_absolute_error(test["coin_in_usd"], preds)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("method", method)
        mlflow.log_param("train_period", "2025-01-01 to 2025-04-30")
        mlflow.log_param("test_period", "2025-05-01 to 2025-06-30")
        mlflow.log_metric("mae_usd", mae)
        mlflow.log_metric("mae_pct_of_mean", mae / test["coin_in_usd"].mean())
    return mae, run.info.run_id

# Baseline: constant training mean
base_val = train["coin_in_usd"].mean()
test["baseline_pred_usd"] = base_val
mae_base, id_base = log_run("baseline_constant_mean", "constant training-period daily mean", test["baseline_pred_usd"])

# Model 1: average by weekday
wd = train.groupby("weekday")["coin_in_usd"].mean()
test["weekday_pred_usd"] = test["weekday"].map(wd)
mae_wd, id_wd = log_run("weekday_average", "average coin-in by weekday (training only)", test["weekday_pred_usd"])

# Model 2: weekday average x uplift for holidays/events (uplifts learned on training only)
normal = train[(train.is_holiday == 0) & (train.is_event == 0)]
wd_normal = normal.groupby("weekday")["coin_in_usd"].mean()
def uplift(flag):
    sub = train[train[flag] == 1]
    if len(sub) < 2:
        return 1.0
    return (sub["coin_in_usd"] / sub["weekday"].map(wd_normal)).mean()
up_hol, up_ev = uplift("is_holiday"), uplift("is_event")
print(f"Holiday uplift: {up_hol:.2f}x | Event uplift: {up_ev:.2f}x")
test["event_pred_usd"] = (test["weekday"].map(wd_normal)
                          * np.where(test.is_holiday == 1, up_hol, 1.0)
                          * np.where(test.is_event == 1, up_ev, 1.0))
mae_ev, id_ev = log_run("weekday_plus_events", "weekday average x holiday/event uplift (training only)", test["event_pred_usd"])

results_tbl = pd.DataFrame({
    "model_name": ["baseline_constant_mean", "weekday_average", "weekday_plus_events"],
    "mae_usd": [mae_base, mae_wd, mae_ev],
    "mlflow_run_id": [id_base, id_wd, id_ev],
})
results_tbl["improvement_vs_baseline_pct"] = (1 - results_tbl["mae_usd"] / mae_base) * 100
display(results_tbl)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Chart: actual vs predictions on the test period

# COMMAND ----------

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(test["gaming_date"], test["coin_in_usd"], label="Actual", linewidth=2)
ax.plot(test["gaming_date"], test["baseline_pred_usd"], label="Baseline (constant mean)", linestyle="--")
ax.plot(test["gaming_date"], test["weekday_pred_usd"], label="Weekday average")
ax.plot(test["gaming_date"], test["event_pred_usd"], label="Weekday + events")
ax.set_title("Daily floor coin-in: May-June 2025 holdout")
ax.set_ylabel("Coin-in (USD)")
ax.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Decision the forecast supports
# MAGIC Which weekdays are busiest? Extra technicians and promotions should go where expected play is highest,
# MAGIC because an hour of downtime costs more on those days.

# COMMAND ----------

wd_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
busy = wd.reindex(wd_order).rename("forecast_daily_coin_in_usd").to_frame()
busy["vs_average_pct"] = (busy["forecast_daily_coin_in_usd"] / busy["forecast_daily_coin_in_usd"].mean() - 1) * 100
display(busy.round(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save tables (contract columns)
# MAGIC `model_pred_usd` = the best non-baseline model on the test MAE.

# COMMAND ----------

best = results_tbl.iloc[1:].sort_values("mae_usd").iloc[0]["model_name"]
pred_col = {"weekday_average": "weekday_pred_usd", "weekday_plus_events": "event_pred_usd"}[best]
print("Best model:", best)

train["baseline_pred_usd"] = base_val
train["weekday_pred_usd"] = train["weekday"].map(wd)
train["event_pred_usd"] = (train["weekday"].map(wd_normal)
                           * np.where(train.is_holiday == 1, up_hol, 1.0)
                           * np.where(train.is_event == 1, up_ev, 1.0))
train["split"], test["split"] = "train", "test"

out = pd.concat([train, test])
forecast_results = pd.DataFrame({
    "gaming_date": out["gaming_date"].dt.date,
    "actual_usd": out["coin_in_usd"],
    "baseline_pred_usd": out["baseline_pred_usd"],
    "model_pred_usd": out[pred_col],
    "split": out["split"],
})
spark.createDataFrame(forecast_results).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("workspace.casino.p3_forecast_results")
spark.createDataFrame(results_tbl[["model_name", "mae_usd", "mlflow_run_id"]]).write.mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable("workspace.casino.p3_model_metrics")
print("Saved p3_forecast_results and p3_model_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Notes for claims (copy these numbers into 04_claims_actions)

# COMMAND ----------

print(f"Baseline MAE:        ${mae_base:,.0f}")
print(f"Weekday MAE:         ${mae_wd:,.0f}  ({(1-mae_wd/mae_base)*100:+.1f}% vs baseline)")
print(f"Weekday+events MAE:  ${mae_ev:,.0f}  ({(1-mae_ev/mae_base)*100:+.1f}% vs baseline)")
print("Busiest forecast day:", busy['forecast_daily_coin_in_usd'].idxmax(),
      "| Quietest:", busy['forecast_daily_coin_in_usd'].idxmin())