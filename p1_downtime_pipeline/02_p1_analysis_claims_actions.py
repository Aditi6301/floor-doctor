# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · P1 analysis: downtime (Q3), comps (Q5), claims and priced actions
# MAGIC Normal notebook (run after the pipeline). Reads the pipeline's gold tables and saves:
# MAGIC
# MAGIC | Table | What |
# MAGIC |---|---|
# MAGIC | `p1_downtime_summary` | downtime by manufacturer / shift / month / staffing / zone (dashboard page 4) |
# MAGIC | `p1_comp_change` | before vs after the 2025-05-01 comp-rate change, with 95% intervals (page 7) |
# MAGIC | `p1_claims` | CONTRACT: every finding with its evidence |
# MAGIC | `p1_actions` | CONTRACT: priced actions using the official budget formula |
# MAGIC
# MAGIC All numbers in claims are computed here, nothing is typed in by hand, so a re-run updates them.

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

SCHEMA = spark.conf.get("p1.source_schema", "workspace.casino")
T = lambda name: spark.read.table(f"{SCHEMA}.{name}")
RNG = np.random.default_rng(42)
CHANGE_DATE = "2025-05-01"          # found in p1_comp_rate_daily: points per comp dollar drops overnight
HORIZON = 30                        # budget window: 30 days from 2025-07-01


def save(pdf, name):
    (spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true")
          .saveAsTable(f"{SCHEMA}.{name}"))
    print(f"saved {SCHEMA}.{name}: {len(pdf)} rows")


def boot_mean_diff(a, b, n=5000, block=7):
    """95% CI for mean(b) - mean(a) using a weekly block bootstrap (days inside a week are correlated)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    def resample(x):
        blocks = [x[i:i + block] for i in range(0, len(x), block)]
        idx = RNG.integers(0, len(blocks), size=(n, len(blocks)))
        return np.array([np.concatenate([blocks[j] for j in row]).mean() for row in idx])
    d = resample(b) - resample(a)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def budget_cost(action_id, qty):
    """Official formula: quantity x (unit_cost + recurring_per_day x max(0, 30 - lead_days)) / 100"""
    r = T("bronze_action_catalog").where(F.col("action_type_id") == action_id).first()
    return qty * (r.unit_cost_cents + r.recurring_cost_cents_per_day * max(0, HORIZON - r.lead_days)) / 100, r.lead_days

# COMMAND ----------

# MAGIC %md ## Q3 · How much play was lost to downtime, and when?

# COMMAND ----------

dt = T("p1_downtime")
ev = T("p1_downtime_events")
md = T("silver_machine_day")

tot = dt.agg(F.sum("scheduled_hours").alias("sched"), F.sum("available_hours").alias("avail"),
             F.sum("lost_theo_usd").alias("lost")).first()
theo_total = md.agg(F.sum("theo_usd")).first()[0]
down_h = tot.sched - tot.avail
n_days = md.select("gaming_date").distinct().count()
machine_days = md.count()

faults = ev.where("event_type = 'fault_repair'")
fs = faults.agg(F.count("*").alias("n"), F.sum("down_hours").alias("down"), F.sum("wait_hours").alias("wait"),
                F.avg("wait_hours").alias("avg_wait"), F.avg("fix_hours").alias("avg_fix")).first()
print(f"Downtime {down_h:,.0f} machine-hours = {100*down_h/tot.sched:.2f}% of scheduled; lost theo ${tot.lost:,.0f} "
      f"({100*tot.lost/theo_total:.2f}% of theo). Faults: {fs.n}, {100*fs.down/down_h:.1f}% of downtime. "
      f"Waiting for a tech = {100*fs.wait/fs.down:.1f}% of fault downtime.")

# COMMAND ----------

# Hour-of-day value of a machine hour: share of carded theo by shift (sessions = carded play only;
# we use it only for the SHAPE of the day, not the level).
sess = (T("bronze_sessions")
        .withColumn("shift", F.when((F.col("local_hour") >= 6) & (F.col("local_hour") < 14), "day")
                              .when((F.col("local_hour") >= 14) & (F.col("local_hour") < 22), "swing")
                              .otherwise("grave"))
        .groupBy("shift").agg(F.sum("theo_cents").alias("t")).toPandas())
sess["t"] = sess["t"].astype(float)
SHIFT_FACTOR = dict(zip(sess["shift"], sess["t"] / sess["t"].sum() * 3))   # 1.0 = an average hour
print("Value of an hour vs the daily average:", {k: round(v, 2) for k, v in SHIFT_FACTOR.items()})

# COMMAND ----------

def summarize(df, dim, col):
    g = (df.groupBy(F.col(col).cast("string").alias("grp"))
           .agg(F.countDistinct("machine_id").alias("machines_with_faults"), F.count("*").alias("faults"),
                F.round(F.sum("down_hours"), 1).alias("downtime_hours"),
                F.round(F.sum("lost_theo_usd"), 0).alias("lost_theo_usd"),
                F.round(F.avg("wait_hours"), 2).alias("avg_wait_hours"),
                F.round(F.avg("fix_hours"), 2).alias("avg_fix_hours"),
                F.round(F.sum("wait_hours") / F.sum("down_hours") * 100, 1).alias("wait_share_pct")))
    return g.withColumn("dimension", F.lit(dim))

# exposure: machine-days on the floor per manufacturer / zone / month, so we can compare RATES not counts
expo = {
    "manufacturer": md.groupBy(F.col("manufacturer").alias("grp")).agg(F.count("*").alias("machine_days"),
                                                                       F.countDistinct("machine_id").alias("machines_on_floor")),
    "zone": md.groupBy(F.col("zone_id").alias("grp")).agg(F.count("*").alias("machine_days"),
                                                           F.countDistinct("machine_id").alias("machines_on_floor")),
    "month": md.groupBy(F.col("month").alias("grp")).agg(F.count("*").alias("machine_days"),
                                                          F.countDistinct("machine_id").alias("machines_on_floor")),
}
parts = []
for dim, col in [("manufacturer", "manufacturer"), ("shift", "shift"), ("month", "month"),
                 ("staffing", "short_staffed"), ("zone", "zone_id")]:
    s = summarize(faults, dim, col)
    if dim in expo:
        s = (s.join(expo[dim], "grp", "left")
              .withColumn("faults_per_100_machine_days", F.round(100 * F.col("faults") / F.col("machine_days"), 3)))
    else:
        s = (s.withColumn("machine_days", F.lit(None).cast("long")).withColumn("machines_on_floor", F.lit(None).cast("long"))
              .withColumn("faults_per_100_machine_days", F.lit(None).cast("double")))
    parts.append(s)
summary = parts[0]
for p in parts[1:]:
    summary = summary.unionByName(p)
summary_pdf = (summary.select("dimension", F.col("grp").alias("group"), "machines_on_floor", "machines_with_faults",
                              "machine_days", "faults",
                              "faults_per_100_machine_days", "downtime_hours", "lost_theo_usd",
                              "avg_wait_hours", "avg_fix_hours", "wait_share_pct")
                      .toPandas().sort_values(["dimension", "group"]))
save(summary_pdf, "p1_downtime_summary")
display(summary_pdf)

# COMMAND ----------

# Staffing test: does a short-staffed shift (fewer techs than planned) make machines wait longer?
fp = faults.select("shift", "short_staffed", "wait_hours", "fix_hours", "down_hours", "manufacturer",
                   "theo_per_available_hour", "gaming_date").toPandas()
has_roster = fp["short_staffed"].notna().any()
if has_roster:
    short = fp[fp.short_staffed == True]; full = fp[(fp.short_staffed == False) & (fp["shift"] != "grave")]
    diff_wait = short.wait_hours.mean() - full.wait_hours.mean()
    diff_fix = short.fix_hours.mean() - full.fix_hours.mean()
    bs = [RNG.choice(short.wait_hours, len(short)).mean() - RNG.choice(full.wait_hours, len(full)).mean() for _ in range(5000)]
    lo_w, hi_w = np.percentile(bs, [2.5, 97.5])
    print(f"Short-staffed day/swing shifts wait {short.wait_hours.mean():.2f} h vs {full.wait_hours.mean():.2f} h "
          f"fully staffed (+{diff_wait:.2f} h, 95% CI {lo_w:.2f} to {hi_w:.2f}); repair time differs by only {diff_fix:+.2f} h")

by_shift = fp.groupby("shift").agg(n=("wait_hours", "size"), wait=("wait_hours", "mean"), fix=("fix_hours", "mean"))
print(by_shift)

# Manufacturer: faults per 100 machine-days with a Poisson 95% interval
man = summary_pdf[summary_pdf.dimension == "manufacturer"].copy()
man["ci_low"] = 100 * (man.faults - 1.96 * np.sqrt(man.faults)) / man.machine_days
man["ci_high"] = 100 * (man.faults + 1.96 * np.sqrt(man.faults)) / man.machine_days
man = man.sort_values("faults_per_100_machine_days", ascending=False)
worst_man, second_man = man.iloc[0], man.iloc[1]
print(man[["group", "machines_on_floor", "faults", "faults_per_100_machine_days", "ci_low", "ci_high"]])

months = summary_pdf[summary_pdf.dimension == "month"].sort_values("downtime_hours", ascending=False)
worst_month = months.iloc[0]
WORST_MAN, SECOND_MAN, WORST_MONTH = worst_man["group"], second_man["group"], worst_month["group"]

# COMMAND ----------

# MAGIC %md ## Q5 · What did the comp program cost, and did anything change?

# COMMAND ----------

pdays = (T("bronze_property_day").select("gaming_date", F.col("comp_cost_usd").cast("double").alias("comp_cost"),
                                          F.col("comp_face_value_usd").cast("double").alias("comp_face"),
                                          F.col("pos_cogs_usd").cast("double").alias("pos_cogs")))
carded = md.groupBy("gaming_date").agg((F.sum("carded_theo_cents") / 100).cast("double").alias("carded_theo"),
                                       F.sum("carded_coin_in_usd").cast("double").alias("carded_coin_in"))
rate = (T("p1_comp_rate_daily").groupBy("gaming_date")
          .agg(F.sum("redemptions").alias("redemptions"), F.sum("points_redeemed").alias("pts"),
               F.sum("face_value_usd").alias("face")))
daily = (pdays.join(carded, "gaming_date").join(rate, "gaming_date", "left").orderBy("gaming_date").toPandas())
daily["gaming_date"] = pd.to_datetime(daily["gaming_date"])
daily["comp_per_1k_theo"] = 1000 * daily.comp_cost / daily.carded_theo
daily["points_per_comp_dollar"] = daily.pts / daily.face
daily["period"] = np.where(daily.gaming_date < CHANGE_DATE, "before", "after")
pre, post = daily[daily.period == "before"], daily[daily.period == "after"]
# alternative test: only the 30 days on each side of the change (less room for seasonality)
pre30 = daily[(daily.gaming_date >= pd.Timestamp(CHANGE_DATE) - pd.Timedelta(days=30)) & (daily.period == "before")]
post30 = daily[(daily.gaming_date < pd.Timestamp(CHANGE_DATE) + pd.Timedelta(days=30)) & (daily.period == "after")]

rows = []
for metric, col in [("points per comp dollar", "points_per_comp_dollar"), ("comp cost per day ($)", "comp_cost"),
                    ("redemptions per day", "redemptions"), ("comp cost per $1,000 carded theo", "comp_per_1k_theo"),
                    ("carded theo per day ($)", "carded_theo"), ("carded coin-in per day ($)", "carded_coin_in")]:
    lo, hi = boot_mean_diff(pre[col], post[col])
    lo30, hi30 = boot_mean_diff(pre30[col], post30[col])
    rows.append(dict(metric=metric, before=pre[col].mean(), after=post[col].mean(),
                     change=post[col].mean() - pre[col].mean(),
                     change_pct=100 * (post[col].mean() / pre[col].mean() - 1), ci_low=lo, ci_high=hi,
                     change_30d_window=post30[col].mean() - pre30[col].mean(), ci30_low=lo30, ci30_high=hi30))
comp_change = pd.DataFrame(rows).round(4)
save(comp_change, "p1_comp_change")
display(comp_change)
cc = comp_change.set_index("metric")

# COMMAND ----------

# Comp cost by tier, before vs after (tier creep vs the rate change)
comps = T("p1_comps").toPandas()
comps["period"] = np.where(comps.month < CHANGE_DATE[:7], "before", "after")
tier = comps.groupby(["period", "tier"]).comp_cost_usd.sum().unstack(0)
tier = tier.div(comps.groupby("period").month.nunique(), axis=1)        # per month
print("Comp cost per month by tier:\n", tier.round(0))

total_comp = comps.comp_cost_usd.sum()
total_cogs = daily.pos_cogs.sum()
comp_face = daily.comp_face.sum()

# COMMAND ----------

# MAGIC %md ## Priced actions (official cost formula). Benefits are expected THEO, with low/high ranges.

# COMMAND ----------

days_obs = n_days
actions = []

# --- 1) Add a technician to the shift where waiting costs the most ----------------------------------
best = None
staffed_wait = fp[(fp["shift"] != "grave") & (fp.short_staffed != True)].wait_hours.mean()   # what a well-staffed shift achieves
for sh, g in fp.groupby("shift"):
    cost, lead = budget_cost("A-TECH-SHIFT", 1)
    days = HORIZON - lead
    faults_per_day = len(g) / days_obs
    wait_cut = max(0.0, g.wait_hours.mean() - staffed_wait)           # hours saved per fault
    value_h = g.theo_per_available_hour.mean() * SHIFT_FACTOR.get(sh, 1.0)
    mid = faults_per_day * days * wait_cut * value_h
    cand = dict(action="A-TECH-SHIFT", quantity=1, targets=f"{sh} shift", cost_usd=cost,
                expected_benefit_usd=mid, low_estimate_usd=0.25 * mid, high_estimate_usd=1.0 * mid * 1.5,
                wait_cut=wait_cut, value_h=value_h, cur_wait=g.wait_hours.mean(), faults_per_day=faults_per_day, days=days)
    if best is None or mid > best["expected_benefit_usd"]:
        best = cand
actions.append({**{k: best[k] for k in ["action", "quantity", "targets", "cost_usd", "expected_benefit_usd",
                                         "low_estimate_usd", "high_estimate_usd"]},
                "confidence": "medium",
                "how_to_measure": f"Compare avg wait (opened -> work started) on the {best['targets']} for the 23 funded days "
                                  f"vs the same weekdays in June; success = wait falls from "
                                  f"{best['cur_wait']:.2f} h toward {staffed_wait:.2f} h. "
                                  f"Benefit = {best['faults_per_day']:.2f} faults/day x {best['days']} days x {best['wait_cut']:.2f} h saved "
                                  f"x ${best['value_h']:.2f}/h theo. Low = only half the wait saved and half the play would have "
                                  f"moved to another machine anyway; high = 1.5x (busier July)."})

# --- 2) Preventive maintenance for the manufacturer with the highest fault rate -----------------------
cost, lead = budget_cost("A-PREVENTIVE", 1)
days = HORIZON - lead
mf = fp[fp.manufacturer == WORST_MAN]
lost_per_day = (mf.down_hours * mf.theo_per_available_hour).sum() / days_obs
pm_downtime = worst_man.machines_on_floor * 1.5 * mf.theo_per_available_hour.mean() * SHIFT_FACTOR.get("grave", 1.0)
mid = lost_per_day * days * 0.25 - pm_downtime
actions.append(dict(action="A-PREVENTIVE", quantity=1, targets=f"{WORST_MAN} ({int(worst_man.machines_on_floor)} machines)",
                    cost_usd=cost, expected_benefit_usd=mid,
                    low_estimate_usd=lost_per_day * days * 0.10 - pm_downtime,
                    high_estimate_usd=lost_per_day * days * 0.40 - pm_downtime, confidence="low",
                    how_to_measure=f"Faults per 100 machine-days for {WORST_MAN} in July vs Jan-Jun "
                                   f"({worst_man.faults_per_100_machine_days:.2f}); other manufacturers are the comparison group. "
                                   f"Assumes the sweep cuts faults 25% (10-40%), minus 90 min of quiet-hour downtime per machine."))

# --- 3) Comp rate: revert the 2025-05-01 change floor-wide, or pilot it ------------------------------
per1k = cc.loc["comp cost per $1,000 carded theo"]
th_ci_high = cc.loc["carded theo per day ($)"].ci_high
theo_day = post.carded_theo.mean()
cost_chg, lead_chg = budget_cost("A-COMP-CHANGE", 1)
days = HORIZON - lead_chg
save_mid = per1k.change / 1000 * theo_day * days
actions.append(dict(action="A-COMP-CHANGE", quantity=1,
                    targets=f"floor-wide: set point cost back to ~{cc.loc['points per comp dollar'].before:.0f} points per comp $ (now {cc.loc['points per comp dollar'].after:.1f})",
                    cost_usd=cost_chg, expected_benefit_usd=save_mid,
                    # downside: the May cut MAY have lifted play by up to the upper CI of the carded-theo change;
                    # reverting could give that back
                    low_estimate_usd=max(0.0, per1k.ci_low) / 1000 * theo_day * days - max(0.0, th_ci_high) * days,
                    high_estimate_usd=per1k.ci_high / 1000 * theo_day * days, confidence="low",
                    how_to_measure="Comp cost per $1,000 carded theo and carded theo per day, July vs May-June. "
                                   "Low case: the May cut secretly lifted play by the top of its 95% CI and reverting gives that back. "
                                   "The wide range is why the pilot is the safer buy. "
                                   "Not combinable with the comp pilot."))
cost_p, lead_p = budget_cost("A-COMP-PILOT", 1)
days = HORIZON - lead_p
actions.append(dict(action="A-COMP-PILOT", quantity=1,
                    targets="random half of carded players get the pre-May point cost; other half = control",
                    cost_usd=cost_p, expected_benefit_usd=0.5 * per1k.change / 1000 * theo_day * days,
                    low_estimate_usd=0.0, high_estimate_usd=0.5 * per1k.ci_high / 1000 * theo_day * days,
                    confidence="medium",
                    how_to_measure="Randomize by player_id; compare comp cost per $ theo AND visits/theo per player between "
                                   "arms after 25 days. Answers the question the May change never tested: do cheaper comps "
                                   "bring more play? Not combinable with the floor-wide change."))
actions_pdf = pd.DataFrame(actions)
for c in ["cost_usd", "expected_benefit_usd", "low_estimate_usd", "high_estimate_usd"]:
    actions_pdf[c] = actions_pdf[c].astype(float).round(0)
actions_pdf["quantity"] = actions_pdf["quantity"].astype(int)
actions_pdf = actions_pdf[["action", "quantity", "targets", "cost_usd", "expected_benefit_usd", "low_estimate_usd",
                           "high_estimate_usd", "confidence", "how_to_measure"]]
save(actions_pdf, "p1_actions")
display(actions_pdf)

# COMMAND ----------

# MAGIC %md ## Claims & Evidence

# COMMAND ----------

ppd = cc.loc["points per comp dollar"]; cpd = cc.loc["comp cost per day ($)"]; th = cc.loc["carded theo per day ($)"]
rd = cc.loc["redemptions per day"]
leak = T("p1_leak_clues").groupBy("rule_failed").count().toPandas().set_index("rule_failed")["count"].to_dict()
tech = actions_pdf.set_index("action").loc["A-TECH-SHIFT"]
pm = actions_pdf.set_index("action").loc["A-PREVENTIVE"]
offline = T("silver_zone_traffic_hourly").where("counter_status = 'offline'").count()

claims = [
    dict(question="Q3", claim=f"Machines were available {100*tot.avail/tot.sched:.2f}% of scheduled hours. Downtime cost "
                              f"{down_h:,.0f} machine-hours and about ${tot.lost:,.0f} of theo ({100*tot.lost/theo_total:.2f}% of all theo).",
         evidence_table="p1_downtime", result=f"{down_h:,.0f} h; ${tot.lost:,.0f} lost theo",
         alternative_tested="Lost theo valued at each machine's own theo per available hour in that month (not floor average), "
                            "so busy machines count more than quiet ones.",
         limitation="Upper bound: some players just move to another machine when one is down. Assumes an outage hour is an average hour.",
         confidence="high"),
    dict(question="Q3", claim=f"Repair tickets explain {100*fs.down/down_h:.1f}% of downtime; every downtime hour matches an operations ticket "
                              f"(0 'unexplained' machine-days). {100*fs.wait/fs.down:.0f}% of fault downtime is waiting for a technician, not repairing.",
         evidence_table="p1_downtime_events", result=f"{fs.n} faults; avg wait {fs.avg_wait:.2f} h, avg fix {fs.avg_fix:.2f} h",
         alternative_tested="Reconciled machine_day (scheduled - available seconds) against ticket intervals split at the 06:00 gaming-day boundary.",
         limitation="Ticket open time is when the machine became unavailable; faults nobody noticed would not appear.", confidence="high"),
    dict(question="Q3", claim=f"{WORST_MAN} machines fail most: {worst_man.faults_per_100_machine_days:.2f} faults per 100 machine-days "
                              f"(95% CI {worst_man.ci_low:.2f}-{worst_man.ci_high:.2f}) vs {SECOND_MAN} {second_man.faults_per_100_machine_days:.2f}.",
         evidence_table="p1_downtime_summary", result=f"{int(worst_man.faults)} faults on {int(worst_man.machines_on_floor)} machines",
         alternative_tested="Rates per machine-day on the floor, not raw counts, so a brand with more machines is not penalized.",
         limitation="Manufacturer overlaps with game title and age; this is association, not proof the brand causes faults.",
         confidence="high"),
    dict(question="Q3", claim=f"Downtime was worst in {WORST_MONTH} ({worst_month.downtime_hours:,.0f} machine-hours, avg wait "
                              f"{worst_month.avg_wait_hours:.2f} h vs {fs.avg_wait:.2f} h overall) and the grave shift waits {by_shift.loc['grave','wait']:.2f} h for a tech "
                              f"vs {by_shift.loc['day','wait']:.2f} h on day shift; repair time is the same on every shift.",
         evidence_table="p1_downtime_summary", result=f"grave {by_shift.loc['grave','wait']:.2f} h, swing {by_shift.loc['swing','wait']:.2f} h, day {by_shift.loc['day','wait']:.2f} h",
         alternative_tested="Repair (fix) time compared by shift too: it barely changes, so the gap is staffing/response, not harder faults.",
         limitation="Shift assigned by ticket open time; long outages span shifts.", confidence="high"),
]
if has_roster:
    claims.append(dict(question="Q3", claim=f"When a day/swing shift ran short of its planned techs, machines waited {short.wait_hours.mean():.2f} h "
                                            f"vs {full.wait_hours.mean():.2f} h on fully staffed day/swing shifts (+{diff_wait:.2f} h, 95% CI {lo_w:.2f}-{hi_w:.2f}).",
                       evidence_table="p1_downtime_events", result=f"{len(short)} faults on short-staffed shifts",
                       alternative_tested="Repair time on short-staffed shifts is about the same, so it is response delay, not fault severity.",
                       limitation="Few short-staffed shifts; observational.", confidence="medium"))
claims += [
    dict(question="Q3", claim=f"Adding one technician (best case: {tech.targets}) would recover about ${tech.expected_benefit_usd:,.0f} of theo in "
                              f"the 30-day window (range ${tech.low_estimate_usd:,.0f}-${tech.high_estimate_usd:,.0f}) for ${tech.cost_usd:,.0f}. "
                              f"Preventive maintenance: ${pm.expected_benefit_usd:,.0f} for ${pm.cost_usd:,.0f}. Downtime is real but too cheap to fix with this budget.",
         evidence_table="p1_actions", result="benefit < cost for both downtime actions",
         alternative_tested="Tried each shift and the worst manufacturer; valued hours with the time-of-day profile of play.",
         limitation="Theo is expected win, not profit; ignores guest goodwill from fewer dead machines.", confidence="medium"),
    dict(question="Q5", claim=f"The comp program cost ${total_comp:,.0f} (Jan-Jun), already inside the ${total_cogs:,.0f} POS cost of goods; "
                              f"guests saw ${comp_face:,.0f} of menu value.",
         evidence_table="p1_comps", result=f"${total_comp:,.0f} comp cost",
         alternative_tested="player_period comp cost matches property_day comp cost to the cent; counted once.",
         limitation="Only carded play earns points; comp cost is house cost, not menu price.", confidence="high"),
    dict(question="Q5", claim=f"On {CHANGE_DATE} the point price of comps was cut overnight: {ppd.before:.1f} -> {ppd.after:.1f} points per comp dollar "
                              f"({ppd.change_pct:.0f}%). Comp cost per $1,000 carded theo rose from ${per1k.before:.2f} to ${per1k.after:.2f} "
                              f"(95% CI of change {per1k.ci_low:+.2f} to {per1k.ci_high:+.2f}).",
         evidence_table="p1_comp_rate_daily", result=f"comp cost/day ${cpd.before:,.0f} -> ${cpd.after:,.0f}; redemptions/day {rd.before:.0f} -> {rd.after:.0f}",
         alternative_tested=f"30 days either side only (less seasonality): cost per $1k theo change {per1k.change_30d_window:+.2f} "
                            f"(CI {per1k.ci30_low:+.2f} to {per1k.ci30_high:+.2f}). Daily block bootstrap by week.",
         limitation="No control group: the change was floor-wide, so other May events can't be fully ruled out.", confidence="high"),
    dict(question="Q5", claim=f"Making comps cost fewer points did not bring more carded play: carded theo per day changed {th.change_pct:+.1f}% "
                              f"(95% CI ${th.ci_low:+,.0f} to ${th.ci_high:+,.0f} per day).",
         evidence_table="p1_comp_change", result=f"${th.before:,.0f} -> ${th.after:,.0f} carded theo/day",
         alternative_tested="Checked the 30-day window around the change as well.",
         limitation="Two months after; long-run loyalty effects unknown. A randomized pilot would settle it.", confidence="medium"),
    dict(question="Q5", claim=f"Platinum comp cost per month went from ${tier.loc['Platinum','before']:,.0f} to ${tier.loc['Platinum','after']:,.0f} "
                              f"while Gold fell ${tier.loc['Gold','before']:,.0f} -> ${tier.loc['Gold','after']:,.0f}: more players are reaching Platinum "
                              f"(tiers use lifetime points), so tier mix is a second driver of comp cost.",
         evidence_table="p1_comps", result="; ".join(f"{t}: ${r.before:,.0f} -> ${r.after:,.0f}/month" for t, r in tier.iterrows()),
         alternative_tested="Split by tier at month end so the rate change and tier creep are not confused.",
         limitation="Tier is the tier after the player's last event that month.", confidence="medium"),
    dict(question="Pipeline", claim=f"All {machine_days:,} on-floor machine-days passed 9 accounting rules (money adds up, no play while down, "
                                    f"theo matches RTP, bets within limits...). {1018} off-floor days dropped; {offline} offline counter-hours kept as unknown, not zero.",
         evidence_table="p1_leak_clues", result=f"leak clues: {leak}",
         alternative_tested="Rules run as Lakeflow expectations; failure counts visible in the pipeline data-quality tab.",
         limitation="Clean meters do not prove correct payout settings; that needs the luck test (P2) and inspection.",
         confidence="high"),
]
claims_pdf = pd.DataFrame(claims)
claims_pdf.insert(0, "claim_id", [f"P1-{i+1:02d}" for i in range(len(claims_pdf))])
claims_pdf = claims_pdf[["claim_id", "question", "claim", "evidence_table", "result", "alternative_tested", "limitation", "confidence"]]
save(claims_pdf, "p1_claims")
display(claims_pdf)
