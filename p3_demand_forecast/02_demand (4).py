# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # P3 · 02 Demand changes (Q2)
# MAGIC **Question:** What changed in demand across the floor over six months, and why?
# MAGIC
# MAGIC **Approach**
# MAGIC 1. Floor trend: coin-in (demand), not house win (house win includes luck)
# MAGIC 2. Separate *more players* from *more machines*: coin-in per active machine and occupancy
# MAGIC 3. Which zones changed (from `prepared_daily_view`)
# MAGIC 4. Traffic sensors (offline hours excluded, not counted as zero)
# MAGIC 5. Events and real Las Vegas visitation/weather as possible explanations
# MAGIC 6. Test a competing explanation for the biggest change

# COMMAND ----------

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

def find_col(cols, must_have, prefer=None):
    hits = [c for c in cols if all(k in c.lower() for k in must_have)]
    if prefer:
        p = [c for c in hits if prefer in c.lower()]
        if p: return p[0]
    return hits[0] if hits else None

# --- Decimal fix: Spark money columns are DECIMAL, which pandas can't chart or do math with.
# This makes every .toPandas() in this notebook convert decimals to normal floats first.
from pyspark.sql.types import DecimalType
from pyspark.sql import functions as _F
_DF = type(spark.range(1))
if not getattr(_DF, "_decimal_fix", False):
    _orig_toPandas = _DF.toPandas
    def _toPandas_clean(self):
        for f in self.schema.fields:
            if isinstance(f.dataType, DecimalType):
                self = self.withColumn(f.name, _F.col(f.name).cast("double"))
        return _orig_toPandas(self)
    _DF.toPandas = _toPandas_clean
    _DF._decimal_fix = True
print("Decimal fix active")

pdays = spark.table("workspace.casino.bronze_property_day").toPandas()
pdays["gaming_date"] = pd.to_datetime(pdays["gaming_date"])
pdays = pdays.sort_values("gaming_date")
pdays["month"] = pdays["gaming_date"].dt.to_period("M").astype(str)
pdays["week"] = pdays["gaming_date"].dt.to_period("W-SUN").dt.start_time
pdays["occupancy"] = pdays["occupied_machine_hours"] / pdays["available_machine_hours"]
pdays["coin_in_per_machine"] = pdays["coin_in_usd"] / pdays["active_machines"]
pdays["event"] = pdays["events_active"].where(pdays["events_active"].notna() & (pdays["events_active"].astype(str) != "null"))
print(len(pdays), "days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Floor trend by month (ratio of sums)

# COMMAND ----------

monthly = pdays.groupby("month").agg(
    days=("gaming_date", "count"),
    coin_in_usd=("coin_in_usd", "sum"),
    theo_usd=("theo_usd", "sum"),
    house_win_usd=("house_win_usd", "sum"),
    occupied_hours=("occupied_machine_hours", "sum"),
    available_hours=("available_machine_hours", "sum"),
    entrance_entries=("entrance_zone_entries_observed", "sum"),
    avg_active_machines=("active_machines", "mean"),
    event_days=("event", lambda s: s.notna().sum()),
).reset_index()
monthly["coin_in_per_day"] = monthly["coin_in_usd"] / monthly["days"]
monthly["occupancy"] = monthly["occupied_hours"] / monthly["available_hours"]
monthly["observed_hold"] = monthly["house_win_usd"] / monthly["coin_in_usd"]
monthly["theo_hold"] = monthly["theo_usd"] / monthly["coin_in_usd"]
monthly["coin_in_per_day_vs_jan_pct"] = (monthly["coin_in_per_day"] / monthly["coin_in_per_day"].iloc[0] - 1) * 100
display(monthly.round(3))

# COMMAND ----------

weekly = pdays.groupby("week").agg(coin_in_usd=("coin_in_usd", "sum"), days=("gaming_date", "count"),
                                   occupancy=("occupancy", "mean")).reset_index()
weekly["coin_in_per_day"] = weekly["coin_in_usd"] / weekly["days"]

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(pdays["gaming_date"], pdays["coin_in_usd"], alpha=0.35, label="Daily coin-in")
ax.plot(weekly["week"] + pd.Timedelta(days=3), weekly["coin_in_per_day"], linewidth=2.5, label="Weekly average per day")
for _, r in pdays[pdays["event"].notna()].iterrows():
    ax.axvspan(r["gaming_date"], r["gaming_date"] + pd.Timedelta(days=1), color="orange", alpha=0.15)
ax.set_title("Floor coin-in, Jan-Jun 2025 (orange = event days)")
ax.set_ylabel("USD"); ax.legend(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. More players, or more machines?
# MAGIC If coin-in per active machine and occupancy move with total coin-in, demand really changed.
# MAGIC If only total coin-in moved, the floor size changed.

# COMMAND ----------

display(monthly[["month", "avg_active_machines", "coin_in_per_day", "occupancy", "entrance_entries"]].round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Events: how much higher is coin-in on event days?
# MAGIC Compared with the same weekday on non-event days, so weekends don't distort it.

# COMMAND ----------

wd_base = pdays[pdays["event"].isna()].groupby("local_weekday")["coin_in_usd"].mean()
ev_days = pdays[pdays["event"].notna()].copy()
ev_days["same_weekday_normal"] = ev_days["local_weekday"].map(wd_base)
ev_days["uplift_pct"] = (ev_days["coin_in_usd"] / ev_days["same_weekday_normal"] - 1) * 100
event_summary = ev_days.groupby("event").agg(days=("gaming_date", "count"),
                                             first_day=("gaming_date", "min"),
                                             avg_uplift_pct=("uplift_pct", "mean")).reset_index().sort_values("first_day")
display(event_summary.round(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Which zones changed? (`prepared_daily_view`, on-floor days only)

# COMMAND ----------

pdv = spark.table("workspace.casino.bronze_prepared_daily_view").filter("location_id IS NOT NULL")
print([c for c in pdv.columns])

# COMMAND ----------

ZONE_COL = find_col(pdv.columns, ["zone"], prefer="id") or find_col(pdv.columns, ["zone"])
# ZONE_COL = "zone_id"   # override if wrong
print("Zone column:", ZONE_COL)

zone_daily = (pdv.groupBy("gaming_date", ZONE_COL)
              .agg(F.sum("coin_in_usd").alias("coin_in_usd"), F.sum("theo_usd").alias("theo_usd"),
                   F.countDistinct("machine_id").alias("machines"))
              .toPandas())
zone_daily["gaming_date"] = pd.to_datetime(zone_daily["gaming_date"])
zone_daily["month"] = zone_daily["gaming_date"].dt.to_period("M").astype(str)

zone_month = zone_daily.groupby([ZONE_COL, "month"]).agg(coin_in_usd=("coin_in_usd", "sum"),
                                                        days=("gaming_date", "nunique"),
                                                        machines=("machines", "mean")).reset_index()
zone_month["coin_in_per_day"] = zone_month["coin_in_usd"] / zone_month["days"]
zone_month["coin_in_per_machine_day"] = zone_month["coin_in_per_day"] / zone_month["machines"]
pivot = zone_month.pivot(index=ZONE_COL, columns="month", values="coin_in_per_machine_day").astype(float)
display(pivot.round(0))

# COMMAND ----------

# Change: Jan-Feb vs May-Jun, per machine-day (fair across zones of different sizes)
early = zone_month[zone_month["month"].isin(["2025-01", "2025-02"])]
late = zone_month[zone_month["month"].isin(["2025-05", "2025-06"])]
def per_machine_day(d):
    return d.groupby(ZONE_COL).apply(lambda g: g["coin_in_usd"].sum() / (g["days"] * g["machines"]).sum())
zone_change = pd.DataFrame({"early_per_machine_day": per_machine_day(early),
                            "late_per_machine_day": per_machine_day(late)})
zone_change["change_pct"] = (zone_change["late_per_machine_day"] / zone_change["early_per_machine_day"] - 1) * 100
floor_early = early["coin_in_usd"].sum() / (early["days"] * early["machines"]).sum()
floor_late = late["coin_in_usd"].sum() / (late["days"] * late["machines"]).sum()
floor_change = (floor_late / floor_early - 1) * 100
zone_change["vs_floor_pts"] = zone_change["change_pct"] - floor_change
print(f"Whole floor change (per machine-day, Jan-Feb -> May-Jun): {floor_change:+.1f}%")
display(zone_change.sort_values("change_pct").round(1))

# COMMAND ----------

fig, ax = plt.subplots(figsize=(13, 4.5))
zw = zone_daily.copy()
zw["week"] = zw["gaming_date"].dt.to_period("W-SUN").dt.start_time
zw = zw.groupby(["week", ZONE_COL]).agg(c=("coin_in_usd", "sum"), m=("machines", "sum")).reset_index()
zw["per_machine_day"] = zw["c"] / zw["m"]
for z, g in zw.groupby(ZONE_COL):
    ax.plot(g["week"], g["per_machine_day"], label=str(z))
ax.set_title("Weekly coin-in per machine-day by zone"); ax.set_ylabel("USD"); ax.legend(ncol=4); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Traffic sensors (offline hours excluded, not zero)
# MAGIC Counts are **entries**, not unique visitors.

# COMMAND ----------

trf = spark.table("workspace.casino.bronze_zone_traffic_hourly")
print(trf.columns)
display(trf.limit(5))

# COMMAND ----------

T_ZONE = find_col(trf.columns, ["zone"], prefer="id") or find_col(trf.columns, ["sensor"])
T_COUNT = find_col(trf.columns, ["entr"]) or find_col(trf.columns, ["count"])
T_STATUS = find_col(trf.columns, ["status"])
T_DATE = find_col(trf.columns, ["gaming_date"])
T_TIME = find_col(trf.columns, ["local"]) or find_col(trf.columns, ["hour"]) or find_col(trf.columns, ["utc"])
# Overrides if needed:
# T_ZONE, T_COUNT, T_STATUS, T_DATE, T_TIME = "zone_id", "entries", "counter_status", "gaming_date", "hour_start_local"
print("zone:", T_ZONE, "| count:", T_COUNT, "| status:", T_STATUS, "| date:", T_DATE, "| time:", T_TIME)
if T_STATUS:
    display(trf.groupBy(T_STATUS).count())

# COMMAND ----------

t = trf
if not T_DATE:
    # gaming day runs 6 a.m. to 6 a.m.: shift local time back 6 hours before taking the date
    t = t.withColumn("gaming_date", F.to_date(F.col(T_TIME) - F.expr("INTERVAL 6 HOURS")))
    T_DATE = "gaming_date"
ok_values = ["ok", "online", "normal", "valid", "active"]
if T_STATUS:
    t = t.withColumn("sensor_ok", F.lower(F.col(T_STATUS)).isin(ok_values) & F.col(T_COUNT).isNotNull())
else:
    t = t.withColumn("sensor_ok", F.col(T_COUNT).isNotNull())

traffic_daily = (t.groupBy(T_DATE, T_ZONE)
                 .agg(F.sum(F.when(F.col("sensor_ok"), F.col(T_COUNT))).alias("entries_observed"),
                      F.avg(F.col("sensor_ok").cast("int")).alias("coverage"))
                 .withColumn("entries_coverage_adjusted",
                             F.when(F.col("coverage") > 0.5, F.col("entries_observed") / F.col("coverage")))
                 .toPandas())
traffic_daily[T_DATE] = pd.to_datetime(traffic_daily[T_DATE])
traffic_daily["month"] = traffic_daily[T_DATE].dt.to_period("M").astype(str)
print("Share of zone-days with <50% sensor coverage (left blank):",
      round((traffic_daily["coverage"] <= 0.5).mean() * 100, 1), "%")
display(traffic_daily.groupby([T_ZONE, "month"])["entries_coverage_adjusted"].mean().unstack().round(0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Real Las Vegas context (context only: describes all LV visitors, not this casino's guests)

# COMMAND ----------

def try_table(names):
    for n in names:
        for t_name in [f"workspace.casino.bronze_{n}"]:
            try:
                return spark.table(t_name).toPandas(), t_name
            except Exception:
                pass
        try:
            return spark.read.parquet(f"/Volumes/workspace/casino/raw/{n}.parquet").toPandas(), n + ".parquet"
        except Exception:
            pass
    return None, None

visit, vsrc = try_table(["city_visitation_monthly"])
weather, wsrc = try_table(["weather_daily"])
print("Visitation:", vsrc, "| Weather:", wsrc)
if visit is not None:
    print(list(visit.columns)); display(visit.head(12))
if weather is not None:
    print(list(weather.columns)); display(weather.head(5))

# COMMAND ----------

context = monthly[["month", "coin_in_per_day", "occupancy"]].copy()
if visit is not None:
    v_month = find_col(visit.columns, ["month"]) or find_col(visit.columns, ["date"]) or find_col(visit.columns, ["period"])
    v_visitors = find_col(visit.columns, ["visitor"]) or find_col(visit.columns, ["visitation"])
    print("visitation month/visitors columns:", v_month, v_visitors)
    if v_month and v_visitors:
        vv = visit[[v_month, v_visitors]].copy()
        vv["month"] = pd.to_datetime(vv[v_month].astype(str)).dt.to_period("M").astype(str)
        context = context.merge(vv[["month", v_visitors]], on="month", how="left")
if weather is not None:
    w_date = find_col(weather.columns, ["date"])
    w_tmax = find_col(weather.columns, ["max"]) or find_col(weather.columns, ["temp"])
    if w_date and w_tmax:
        ww = weather[[w_date, w_tmax]].copy()
        ww["month"] = pd.to_datetime(ww[w_date]).dt.to_period("M").astype(str)
        context = context.merge(ww.groupby("month")[w_tmax].mean().rename("avg_max_temp").reset_index(), on="month", how="left")
display(context.round(2))
if context.shape[1] > 3:
    display(context.drop(columns="month").corr().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Competing explanation test
# MAGIC For the zone with the biggest drop (or rise), check:
# MAGIC - **Citywide/floor-wide demand?** Did the whole floor move the same way? (`vs_floor_pts` near 0 means yes)
# MAGIC - **Luck?** We use coin-in, not house win, so luck in payouts can't explain it.
# MAGIC - **Fewer machines?** We use per machine-day, so floor size can't explain it.
# MAGIC - **Downtime?** Compare with zone availability (below).
# MAGIC - **Moves/replacements?** Check `placements` for machines entering/leaving the zone.

# COMMAND ----------

biggest = zone_change["vs_floor_pts"].abs().idxmax()
print("Zone that moved most vs the floor:", biggest, "|", zone_change.loc[biggest].round(1).to_dict())

avail_col = find_col(pdv.columns, ["available", "hour"])
if avail_col:
    avail = (pdv.groupBy(ZONE_COL, F.date_format("gaming_date", "yyyy-MM").alias("month"))
             .agg(F.sum(avail_col).alias("available_hours"), F.countDistinct("machine_id").alias("machines"),
                  F.countDistinct("gaming_date").alias("days"))
             .toPandas())
    avail["avail_hours_per_machine_day"] = avail["available_hours"] / (avail["machines"] * avail["days"])
    display(avail[avail[ZONE_COL] == biggest].sort_values("month").round(2))

pl = spark.table("workspace.casino.bronze_placements").toPandas()
print("placements columns:", list(pl.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save `p3_demand` (contract columns)
# MAGIC One row per gaming_date x zone. `traffic_count` = coverage-adjusted entries if the traffic zone names match the floor zone names, else blank.

# COMMAND ----------

ev_map = pdays.set_index("gaming_date")["event"]
demand = zone_daily[["gaming_date", ZONE_COL, "coin_in_usd"]].rename(columns={ZONE_COL: "zone_id"}).copy()
tr = traffic_daily.rename(columns={T_DATE: "gaming_date", T_ZONE: "zone_id"})[["gaming_date", "zone_id", "entries_coverage_adjusted"]]
tr["zone_id"] = tr["zone_id"].astype(str)
demand["zone_id"] = demand["zone_id"].astype(str)
demand = demand.merge(tr, on=["gaming_date", "zone_id"], how="left").rename(columns={"entries_coverage_adjusted": "traffic_count"})
print("Rows with traffic matched:", demand["traffic_count"].notna().mean().round(3) * 100, "%")
demand["event_name"] = demand["gaming_date"].map(ev_map)
demand["gaming_date"] = demand["gaming_date"].dt.date
demand = demand[["gaming_date", "zone_id", "coin_in_usd", "traffic_count", "event_name"]]
spark.createDataFrame(demand).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.casino.p3_demand")

# Extra helper tables for the dashboard (allowed: adding tables is fine)
spark.createDataFrame(monthly.astype({"month": str})).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("workspace.casino.p3_demand_monthly")
zc = zone_change.reset_index().rename(columns={ZONE_COL: "zone_id"}); zc["zone_id"] = zc["zone_id"].astype(str)
spark.createDataFrame(zc).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.casino.p3_zone_change")
spark.createDataFrame(event_summary.assign(first_day=event_summary["first_day"].dt.date)).write.mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable("workspace.casino.p3_event_uplift")
print("Saved p3_demand, p3_demand_monthly, p3_zone_change, p3_event_uplift")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Numbers for claims

# COMMAND ----------

first, last = monthly.iloc[0], monthly.iloc[-1]
print(f"Coin-in per day: {first['month']} ${first['coin_in_per_day']:,.0f} -> {last['month']} ${last['coin_in_per_day']:,.0f} "
      f"({(last['coin_in_per_day']/first['coin_in_per_day']-1)*100:+.1f}%)")
print(f"Occupancy: {first['occupancy']:.1%} -> {last['occupancy']:.1%}")
print("Peak month:", monthly.loc[monthly['coin_in_per_day'].idxmax(), 'month'],
      "| Low month:", monthly.loc[monthly['coin_in_per_day'].idxmin(), 'month'])
print(f"Floor change per machine-day Jan-Feb -> May-Jun: {floor_change:+.1f}%")
print("Zone that moved most vs floor:", biggest, f"{zone_change.loc[biggest, 'change_pct']:+.1f}%")
if len(event_summary):
    top = event_summary.sort_values("avg_uplift_pct", ascending=False).iloc[0]
    print(f"Biggest event uplift: {top['event']} {top['avg_uplift_pct']:+.1f}% vs same weekday")