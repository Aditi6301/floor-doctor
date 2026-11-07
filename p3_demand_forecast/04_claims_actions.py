# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///


# COMMAND ----------

# MAGIC %md
# MAGIC # P3 · 04 Claims and actions
# MAGIC Run after 01, 02 and 03. Numbers are pulled from your saved tables; **edit the text** so each claim
# MAGIC says what you actually found, including the alternative you tested and the limitation.

# COMMAND ----------

import pandas as pd
import numpy as np

fr = spark.table("workspace.casino.p3_forecast_results").toPandas()
mm = spark.table("workspace.casino.p3_model_metrics").toPandas()
conc = spark.table("workspace.casino.p3_concentration").toPandas()
display(mm)

mae_base = mm.loc[mm.model_name == "baseline_constant_mean", "mae_usd"].iloc[0]
best = mm[mm.model_name != "baseline_constant_mean"].sort_values("mae_usd").iloc[0]
top1 = conc.loc[conc.top_pct == 1, "share_of_net_loss"].iloc[0]
top10 = conc.loc[conc.top_pct == 10, "share_of_net_loss"].iloc[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Claims (edit text in quotes; keep the columns)

# COMMAND ----------

claims = pd.DataFrame([
    {"claim_id": "P3-01", "question": "Model",
     "claim": f"The {best.model_name} forecast {'beats' if best.mae_usd < mae_base else 'does not beat'} the constant-mean baseline on the May-June holdout",
     "evidence_table": "p3_model_metrics",
     "result": f"MAE ${best.mae_usd:,.0f} vs baseline ${mae_base:,.0f} ({(1-best.mae_usd/mae_base)*100:+.1f}%)",
     "alternative_tested": "Constant training-period mean (baseline); weekday-only vs weekday+events",
     "limitation": "Only 4 months of training data; one season",
     "confidence": "High"},
    {"claim_id": "P3-02", "question": "Model",
     "claim": "EDIT: Busiest forecast weekdays should get extra technician coverage and any promotions",
     "evidence_table": "p3_forecast_results",
     "result": "EDIT: e.g. Saturdays forecast X% above average daily coin-in",
     "alternative_tested": "Compared weekday forecast against realized May-June weekday averages",
     "limitation": "Weekday pattern may shift with events",
     "confidence": "Medium"},
    {"claim_id": "P3-03", "question": "Q2",
     "claim": "EDIT: Biggest demand change over six months and why",
     "evidence_table": "p3_demand",
     "result": "EDIT: e.g. Avg daily coin-in fell X% from Jan to Jun",
     "alternative_tested": "EDIT: city visitation trend / zone share test from notebook 02",
     "limitation": "LVCVA data describes all Las Vegas visitors, not this casino",
     "confidence": "Medium"},
    {"claim_id": "P3-04", "question": "Q2",
     "claim": "EDIT: Zone with the biggest change, and whether it was zone-specific or floor-wide",
     "evidence_table": "p3_demand",
     "result": "EDIT: e.g. Zone C coin-in per machine-day -X%; share fell Y points",
     "alternative_tested": "Zone share of floor coin-in (rules out floor-wide decline)",
     "limitation": "Traffic sensors offline on some days; entries are not unique visitors",
     "confidence": "Medium"},
    {"claim_id": "P3-05", "question": "Q2",
     "claim": "EDIT: Strongest event lift",
     "evidence_table": "p3_demand",
     "result": "EDIT: e.g. Event X days averaged Y% above a normal same weekday",
     "alternative_tested": "Compared with same weekday on non-event days",
     "limitation": "Few event days; overlapping holidays",
     "confidence": "Medium"},
    {"claim_id": "P3-06", "question": "Q4",
     "claim": "Carded player losses are highly concentrated",
     "evidence_table": "p3_concentration",
     "result": f"Top 1% of carded players = {top1:.0%} of net loss; top 10% = {top10:.0%}",
     "alternative_tested": "Checked by tier and age band to see if one group drives it",
     "limitation": "Carded play only; uncarded players are not visible",
     "confidence": "High"},
    {"claim_id": "P3-07", "question": "Q4",
     "claim": "EDIT: Guest-group finding (tier / age / region)",
     "evidence_table": "p3_players",
     "result": "EDIT",
     "alternative_tested": "EDIT",
     "limitation": "Starting tier may differ from current tier",
     "confidence": "Medium"},
])
spark.createDataFrame(claims).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("workspace.casino.p3_claims")
display(claims)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Action costs from the catalog
# MAGIC `cost = quantity x (unit_cost_cents + recurring_cost_cents_per_day x max(0, 30 - lead_days)) / 100`

# COMMAND ----------

cat = spark.table("workspace.casino.bronze_action_catalog").toPandas()
display(cat)

def find_col(cols, must_have):
    hits = [c for c in cols if all(k in c.lower() for k in must_have)]
    return hits[0] if hits else None

NAME = find_col(cat.columns, ["name"]) or find_col(cat.columns, ["action"])
UNIT = find_col(cat.columns, ["unit_cost"])
REC = find_col(cat.columns, ["recurring"])
LEAD = find_col(cat.columns, ["lead"])
print("Guesses -> name:", NAME, "| unit:", UNIT, "| recurring:", REC, "| lead:", LEAD)

def action_cost(keyword, qty=1):
    row = cat[cat[NAME].astype(str).str.lower().str.contains(keyword.lower())].iloc[0]
    rec = row[REC] if REC and pd.notna(row[REC]) else 0
    lead = row[LEAD] if LEAD and pd.notna(row[LEAD]) else 0
    return qty * (row[UNIT] + rec * max(0, 30 - lead)) / 100, str(row[NAME]), int(lead)

for kw in ["promotion", "sign", "survey"]:
    try:
        print(kw, "->", action_cost(kw))
    except Exception as e:
        print(kw, "-> not matched; check the catalog names", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Benefit estimates
# MAGIC Promotion benefit = event uplift learned in notebook 01/02 x a normal weekend day's coin-in x hold x days.
# MAGIC Keep estimates honest and wide; the team will pick the final plan together.

# COMMAND ----------

train = fr[fr.split == "train"].copy()
train["gaming_date"] = pd.to_datetime(train["gaming_date"])
weekend_coin = train[train.gaming_date.dt.dayofweek >= 5]["actual_usd"].mean()

HOLD = 0.08          # EDIT: use floor theo / coin-in from property_day if available
EVENT_UPLIFT = 0.10  # EDIT: use the event lift from notebook 02 (e.g. 0.12 = +12%)
promo_days = 2

promo_cost, promo_name, promo_lead = action_cost("promotion")
promo_benefit = weekend_coin * EVENT_UPLIFT * HOLD * promo_days
sign_cost, sign_name, _ = action_cost("sign")
survey_cost, survey_name, _ = action_cost("survey")
print(f"Promotion: cost ${promo_cost:,.0f}, est. extra house win ${promo_benefit:,.0f}")

actions = pd.DataFrame([
    {"action": promo_name, "quantity": 1, "targets": "One weekend, busiest forecast days",
     "cost_usd": promo_cost, "expected_benefit_usd": promo_benefit,
     "low_estimate_usd": promo_benefit * 0.3, "high_estimate_usd": promo_benefit * 1.8,
     "confidence": "Low",
     "how_to_measure": "Compare promo weekend coin-in with the weekday forecast; same weekends last month as control"},
    {"action": sign_name, "quantity": 1, "targets": "EDIT: weakest zone from notebook 02",
     "cost_usd": sign_cost, "expected_benefit_usd": 0.0, "low_estimate_usd": 0.0, "high_estimate_usd": 0.0,
     "confidence": "Low",
     "how_to_measure": "Zone share of floor coin-in and traffic entries, 30 days before vs after"},
    {"action": survey_name, "quantity": 1, "targets": "Carded guests, all tiers",
     "cost_usd": survey_cost, "expected_benefit_usd": 0.0, "low_estimate_usd": 0.0, "high_estimate_usd": 0.0,
     "confidence": "Medium",
     "how_to_measure": "Response count and whether it explains the demand change (information value, not revenue)"},
])
spark.createDataFrame(actions).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("workspace.casino.p3_actions")
display(actions)
print("Saved p3_claims and p3_actions")