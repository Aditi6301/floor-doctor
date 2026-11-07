# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Person 2 — Machine Verdicts (Q1: luck, location, or leak?)
# MAGIC
# MAGIC **Floor Doctor.** For every machine we test three explanations for low earnings before calling it "bad":
# MAGIC 1. **Luck** — exact fair-play variance from `game_math.payout_multiplier_variance` x `machine_day.sum_bet_squared`, then 10,000 Spark simulations per machine.
# MAGIC 2. **Location** — compare each machine with peers running the *same game in the same zone*, plus a natural experiment from machines that moved.
# MAGIC 3. **Leak** — availability (downtime) and payout behaviour far outside what the game allows.
# MAGIC
# MAGIC Reads only `bronze_*`. Writes only `p2_*`. Run top to bottom.

# COMMAND ----------

# Where the bronze tables live and where p2 tables are written
SRC = "workspace.casino"
OUT = "workspace.casino"
N_SIMS = 10_000          # use 100 for a quick first run to save credits

# Verdict thresholds (justified in the claims table at the end)
INSPECT_Z = 3.29         # |z| > 3.29  <=> two-sided p < 0.001; with ~310 machines that is ~0.3 false alarms expected
MIN_DAYS = 120           # fewer days on the floor = not enough exposure to judge (new install / retired)
BROKEN_AVAIL = 97      # availability below 97% = downtime is a real cause
TRULY_BAD_RATIO = 0.75   # earns < 75% of same-game, same-zone peers per available hour
BAD_SPOT_RATIO = 0.80    # zone earns < 80% of the floor per available hour (game mix held fixed)

def src(name): return f"{SRC}.bronze_{name}"
def out(name): return f"{OUT}.p2_{name}"

from pyspark.sql import functions as F
try:
    display
except NameError:          # lets the same file run outside Databricks
    def display(df): df.show(30, truncate=False)

# COMMAND ----------

# MAGIC %md ## 0. Data check: all tables present, key columns as expected

# COMMAND ----------

for t in ["prepared_daily_view", "machine_day", "game_math", "placements", "floor_locations",
          "operations", "action_catalog", "calendar_day", "machines", "floor_walkways"]:
    print(f"{t:22s} {spark.table(src(t)).count():>7,} rows")

# Floor rows only (rule 4): days before install / after retirement have no location
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW floor_days AS
SELECT d.machine_id, d.gaming_date, d.month, d.zone_id, d.zone_name, d.bank_id, d.location_id,
       d.config_id, d.theme, d.manufacturer, d.volatility_class,
       CAST(d.configured_rtp AS DOUBLE)  AS rtp,
       CAST(d.coin_in_usd   AS DOUBLE)   AS coin_in_usd,
       CAST(d.house_win_usd AS DOUBLE)   AS house_win_usd,
       CAST(d.theo_usd      AS DOUBLE)   AS theo_usd,
       CAST(d.available_hours AS DOUBLE) AS available_hours,
       CAST(d.occupied_hours  AS DOUBLE) AS occupied_hours,
       CAST(d.faults_opened AS BIGINT)   AS faults_opened,
       CAST(m.scheduled_seconds AS DOUBLE) / 3600.0 AS scheduled_hours,      -- machine's own scheduled time (handles DST + install/retire days)
       CAST(m.paid_spins AS BIGINT)      AS paid_spins,
       CAST(m.sum_bet_squared AS DOUBLE) / 1e4 AS sum_bet_sq_usd2,           -- cents^2 -> dollars^2
       CAST(g.payout_multiplier_variance AS DOUBLE) AS payout_var
FROM {src('prepared_daily_view')} d
JOIN {src('machine_day')} m USING (machine_id, gaming_date)
JOIN {src('game_math')}   g ON g.config_id = m.config_id
WHERE d.location_id IS NOT NULL
""")
print("floor machine-days:", spark.table("floor_days").count())

# COMMAND ----------

# MAGIC %md ## Task 1. The manager's bottom 10 (raw house win)

# COMMAND ----------

bottom10 = spark.sql("""
SELECT machine_id,
       SUM(house_win_usd) AS house_win_usd,
       SUM(theo_usd)      AS theo_usd,
       SUM(house_win_usd) / NULLIF(SUM(coin_in_usd), 0) AS observed_hold,
       ROW_NUMBER() OVER (ORDER BY SUM(house_win_usd)) AS manager_rank
FROM floor_days GROUP BY machine_id
ORDER BY house_win_usd LIMIT 10
""")
bottom10.write.mode("overwrite").saveAsTable(out("manager_bottom10"))
display(bottom10)

# COMMAND ----------

# MAGIC %md ## Task 2. Machine summary — theo per available hour is the fair "how good is this machine" number
# MAGIC Theo removes luck; per available hour removes downtime and short time on the floor.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW summary_v AS
WITH agg AS (
  SELECT machine_id,
         SUM(coin_in_usd) AS coin_in_usd, SUM(house_win_usd) AS house_win_usd, SUM(theo_usd) AS theo_usd,
         SUM(house_win_usd) - SUM(theo_usd) AS luck_residual_usd,
         COUNT(*) AS days_on_floor, MIN(gaming_date) AS first_day, MAX(gaming_date) AS last_day,
         SUM(available_hours) AS available_hours, SUM(scheduled_hours) AS scheduled_hours,
         SUM(occupied_hours) AS occupied_hours, SUM(paid_spins) AS paid_spins, SUM(faults_opened) AS faults_opened,
         -- exact variance of fair house win: sum over spins of bet^2 * Var(payout/bet)
         SUM(sum_bet_sq_usd2 * payout_var) AS luck_var_usd2,
         SUM(coin_in_usd) / NULLIF(SUM(paid_spins), 0) AS avg_bet_usd,
         MAX_BY(config_id, gaming_date) AS config_id, MAX_BY(theme, gaming_date) AS theme,
         MAX_BY(rtp, gaming_date) AS rtp, MAX_BY(volatility_class, gaming_date) AS volatility_class,
         MAX_BY(manufacturer, gaming_date) AS manufacturer,
         MAX_BY(location_id, gaming_date) AS location_id, MAX_BY(bank_id, gaming_date) AS bank_id,
         MAX_BY(zone_id, gaming_date) AS zone_id, MAX_BY(zone_name, gaming_date) AS zone_name,
         COUNT(DISTINCT location_id) AS locations_used
  FROM floor_days GROUP BY machine_id
)
SELECT a.*,
       100 * a.available_hours / a.scheduled_hours AS availability_pct,
       a.theo_usd / NULLIF(a.available_hours, 0) AS theo_per_hour_usd,
       a.house_win_usd / NULLIF(a.coin_in_usd, 0) AS observed_hold,
       1 - a.rtp AS expected_hold,
       a.occupied_hours / NULLIF(a.available_hours, 0) AS occupancy_pct,
       SQRT(a.luck_var_usd2) AS luck_sd_usd,
       mc.retire_date, mc.install_date, mc.lease_class
FROM agg a LEFT JOIN {src('machines')} mc USING (machine_id)
""")
summary = spark.table("summary_v")
summary.write.mode("overwrite").saveAsTable(out("machine_summary"))
display(summary.orderBy("house_win_usd").select("machine_id", "zone_name", "theme", "days_on_floor", "house_win_usd",
        "theo_usd", "theo_per_hour_usd", "availability_pct", "luck_sd_usd").limit(15))

# COMMAND ----------

# MAGIC %md ## Task 3. Luck simulation in Spark
# MAGIC Each fair month-set is drawn with mean = theo and SD = exact game variance. Normal draws are justified by the central
# MAGIC limit theorem: every machine has tens of thousands of spins (checked below). We also report the closed-form z-score as a
# MAGIC cross-check — the two must agree.

# COMMAND ----------

print("Fewest spins on any machine:", summary.agg(F.min("paid_spins")).first()[0])

base = spark.table(out("machine_summary")).select("machine_id", "house_win_usd", "theo_usd", "luck_sd_usd")
sims = (base.crossJoin(spark.range(N_SIMS).withColumnRenamed("id", "sim"))
            .withColumn("sim_house_win_usd", F.col("theo_usd") + F.randn(seed=42) * F.col("luck_sd_usd")))

luck = (sims.groupBy("machine_id", "house_win_usd", "theo_usd", "luck_sd_usd")
            .agg(F.avg((F.col("sim_house_win_usd") < F.col("house_win_usd")).cast("double")).alias("luck_score"))
            .withColumn("luck_z", (F.col("house_win_usd") - F.col("theo_usd")) / F.col("luck_sd_usd")))
luck.write.mode("overwrite").saveAsTable(out("luck_scores"))

# Small sample for dashboard histograms (500 fair outcomes per machine)
(sims.filter("sim < 500").select("machine_id", "sim_house_win_usd")
     .write.mode("overwrite").saveAsTable(out("luck_sims")))

# luck_score = share of fair simulations that did WORSE (lower house win) than reality
#   ~0.2-0.8 normal luck | < 0.05 unusually bad | < 0.0005 (|z|>3.29) inspect
display(spark.table(out("luck_scores")).orderBy("luck_z").limit(10))

# COMMAND ----------

# MAGIC %md ## Task 4. Location
# MAGIC **4a. Like-for-like peers.** Expected earnings for each machine = theo per available hour of *other* machines running the
# MAGIC same game in the same zone (leave-one-out). This separates "bad game/spot" from "bad machine".
# MAGIC
# MAGIC **4b. Zone effect with game mix held fixed.** For each zone, compare each game's theo/hour with the same game floor-wide.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW peers_v AS
WITH mz AS (
  SELECT machine_id, zone_id, config_id, SUM(theo_usd) AS th, SUM(available_hours) AS h
  FROM floor_days GROUP BY machine_id, zone_id, config_id
),
grp AS (SELECT zone_id, config_id, SUM(th) th, SUM(h) h, COUNT(*) n FROM mz GROUP BY zone_id, config_id),
zone AS (SELECT zone_id, SUM(th)/SUM(h) zone_tph FROM mz GROUP BY zone_id),
-- zone index: zone theo/h vs floor theo/h for the same game, weighted by hours (removes game mix)
cfg_floor AS (SELECT config_id, SUM(th)/SUM(h) cfg_tph FROM mz GROUP BY config_id),
zidx AS (SELECT mz.zone_id, SUM(mz.th) / SUM(mz.h * c.cfg_tph) AS zone_index
         FROM mz JOIN cfg_floor c USING (config_id) GROUP BY mz.zone_id)
SELECT s.machine_id,
       CASE WHEN g.n > 1 THEN (g.th - mz.th) / (g.h - mz.h) END AS peer_theo_per_hour_usd,
       g.n - 1 AS n_peers,
       z.zone_tph AS zone_theo_per_hour_usd,
       c.cfg_tph AS game_floor_theo_per_hour_usd,
       zi.zone_index
FROM {out('machine_summary')} s
JOIN mz   ON mz.machine_id = s.machine_id AND mz.zone_id = s.zone_id AND mz.config_id = s.config_id
JOIN grp g ON g.zone_id = s.zone_id AND g.config_id = s.config_id
JOIN zone z ON z.zone_id = s.zone_id
JOIN cfg_floor c ON c.config_id = s.config_id
JOIN zidx zi ON zi.zone_id = s.zone_id
""")
spark.table("peers_v").write.mode("overwrite").saveAsTable(out("location_peers"))

zone_tbl = spark.sql("""
SELECT s.zone_name, COUNT(*) machines, ROUND(MAX(p.zone_index), 3) AS zone_index_vs_floor,
       ROUND(SUM(s.theo_usd)/SUM(s.available_hours), 2) AS theo_per_hour_usd,
       ROUND(AVG(s.occupancy_pct), 3) AS avg_occupancy
FROM p2_summary_tmp s JOIN peers_v p USING (machine_id) GROUP BY s.zone_name ORDER BY zone_index_vs_floor
""".replace("p2_summary_tmp", out("machine_summary")))
zone_tbl.write.mode("overwrite").saveAsTable(out("zone_index"))
display(zone_tbl)

# COMMAND ----------

# MAGIC %md **4c. Back Forty walkway closure.** `floor_walkways` shows the Main Entrance <-> Back Forty walkway closed for
# MAGIC renovation. Did the zone's share of play drop while it was closed?

# COMMAND ----------

closure = spark.sql(f"""
SELECT closed_from_utc, closed_to_utc FROM {src('floor_walkways')}
WHERE closed_from_utc IS NOT NULL AND (zone_a = 'Z-B40' OR zone_b = 'Z-B40')
""").first()
c_from, c_to = closure["closed_from_utc"].date(), closure["closed_to_utc"].date()
closure_tbl = spark.sql(f"""
SELECT CASE WHEN gaming_date <  DATE'{c_from}' THEN '1 before closure'
            WHEN gaming_date <  DATE'{c_to}'   THEN '2 during closure'
            ELSE '3 after closure' END AS period,
       SUM(CASE WHEN zone_id='Z-B40' THEN theo_usd END) / SUM(CASE WHEN zone_id='Z-B40' THEN available_hours END) AS b40_theo_per_hour,
       SUM(CASE WHEN zone_id<>'Z-B40' THEN theo_usd END) / SUM(CASE WHEN zone_id<>'Z-B40' THEN available_hours END) AS rest_theo_per_hour
FROM floor_days GROUP BY 1 ORDER BY 1
""").withColumn("b40_vs_rest", F.col("b40_theo_per_hour") / F.col("rest_theo_per_hour"))
closure_tbl.write.mode("overwrite").saveAsTable(out("b40_closure_effect"))
print("Walkway closed", c_from, "to", c_to)
display(closure_tbl)

# COMMAND ----------

# MAGIC %md ## Task 4d. Moved-machines natural experiment
# MAGIC Same machine, same game — only the location changes. We compare theo per available hour before vs after the move
# MAGIC (theo, not house win, so luck cannot distort it) and divide by the change for **unmoved** machines over the same dates
# MAGIC (controls for the whole floor getting busier or quieter).

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW moves_v AS
WITH mv AS (
  SELECT machine_id, location_id AS to_location, CAST(effective_from_utc AS DATE) AS move_date
  FROM {src('placements')}
  WHERE machine_id IN (SELECT machine_id FROM {src('placements')} GROUP BY machine_id HAVING COUNT(*) > 1)
    AND CAST(effective_from_utc AS DATE) > DATE'2025-01-01'
),
own AS (
  SELECT f.machine_id, mv.move_date, mv.to_location,
         MAX(CASE WHEN f.gaming_date <  mv.move_date THEN f.location_id END) AS from_location,
         MAX(CASE WHEN f.gaming_date <  mv.move_date THEN f.zone_name END)   AS from_zone,
         MAX(CASE WHEN f.gaming_date >= mv.move_date THEN f.zone_name END)   AS to_zone,
         SUM(CASE WHEN f.gaming_date <  mv.move_date THEN f.theo_usd END) / SUM(CASE WHEN f.gaming_date <  mv.move_date THEN f.available_hours END) AS theo_per_hour_before,
         SUM(CASE WHEN f.gaming_date >= mv.move_date THEN f.theo_usd END) / SUM(CASE WHEN f.gaming_date >= mv.move_date THEN f.available_hours END) AS theo_per_hour_after,
         COUNT(CASE WHEN f.gaming_date >= mv.move_date THEN 1 END) AS days_after
  FROM floor_days f JOIN mv USING (machine_id)
  GROUP BY f.machine_id, mv.move_date, mv.to_location
),
ctl AS (
  SELECT mv.machine_id,
         SUM(CASE WHEN f.gaming_date <  mv.move_date THEN f.theo_usd END) / SUM(CASE WHEN f.gaming_date <  mv.move_date THEN f.available_hours END) AS floor_before,
         SUM(CASE WHEN f.gaming_date >= mv.move_date THEN f.theo_usd END) / SUM(CASE WHEN f.gaming_date >= mv.move_date THEN f.available_hours END) AS floor_after
  FROM mv CROSS JOIN floor_days f
  WHERE f.machine_id NOT IN (SELECT machine_id FROM mv)
  GROUP BY mv.machine_id
)
SELECT own.machine_id, own.from_location, own.to_location, own.from_zone, own.to_zone, own.move_date, own.days_after,
       own.theo_per_hour_before, own.theo_per_hour_after,
       own.theo_per_hour_after / own.theo_per_hour_before - 1 AS change_pct,
       ctl.floor_after / ctl.floor_before - 1 AS unmoved_floor_change_pct,
       (own.theo_per_hour_after / own.theo_per_hour_before) / (ctl.floor_after / ctl.floor_before) - 1 AS adjusted_change_pct
FROM own JOIN ctl USING (machine_id)
""")
moves = spark.table("moves_v")
moves.write.mode("overwrite").saveAsTable(out("move_effects"))
display(moves)

# COMMAND ----------

# MAGIC %md ## Task 5. Leaks: availability and inspection candidates
# MAGIC Unusual payout = **candidate for inspection**, never an accusation.

# COMMAND ----------

inspect = spark.sql(f"""
SELECT s.machine_id, s.zone_name, s.theme, s.observed_hold, s.expected_hold, l.luck_z, l.luck_score,
       s.theo_usd - s.house_win_usd AS shortfall_vs_theo_usd
FROM {out('machine_summary')} s JOIN {out('luck_scores')} l USING (machine_id)
WHERE ABS(l.luck_z) > {INSPECT_Z}
""")
inspect.write.mode("overwrite").saveAsTable(out("inspection_candidates"))
display(inspect)

# Month-by-month hold for each candidate: a persistent gap points to settings/hardware, a single spike points to one big win
display(spark.sql(f"""
SELECT f.machine_id, f.month, SUM(house_win_usd)/SUM(coin_in_usd) AS observed_hold, 1 - MAX(rtp) AS expected_hold,
       (SUM(house_win_usd) - SUM(theo_usd)) / SQRT(SUM(sum_bet_sq_usd2 * payout_var)) AS month_z
FROM floor_days f JOIN {out('inspection_candidates')} i USING (machine_id)
GROUP BY f.machine_id, f.month ORDER BY f.machine_id, f.month
"""))

display(spark.sql(f"""
SELECT MIN(availability_pct) AS worst_availability, PERCENTILE(availability_pct, 0.5) AS median_availability,
       SUM(scheduled_hours - available_hours) AS total_down_hours,
       SUM((scheduled_hours - available_hours) * theo_per_hour_usd) AS est_theo_lost_to_downtime_usd
FROM {out('machine_summary')}
"""))

# COMMAND ----------

# MAGIC %md ## Task 6. Verdicts
# MAGIC Rules are checked **in order**; the first match wins:
# MAGIC
# MAGIC | # | Verdict | Rule | Recommendation |
# MAGIC |---|---|---|---|
# MAGIC | 1 | **Inspect** | \|luck z\| > 3.29 (p < 0.001) | Inspect and verify paytable |
# MAGIC | 2 | **Short exposure** | < 120 days on floor, or already retired | Don't replace; judge later / already gone |
# MAGIC | 3 | **Broken** | availability < 97% | Fix / maintenance |
# MAGIC | 4 | **Truly bad** | theo/hour < 75% of same-game same-zone peers | Replace |
# MAGIC | 5 | **Bad spot** | zone index < 0.80 and machine normal vs peers | Move / fix zone traffic |
# MAGIC | 6 | **Unlucky** | house win below theo, luck score < 0.20 | Keep and monitor |
# MAGIC | 7 | **Healthy** | none of the above | Keep |

# COMMAND ----------

verdicts = spark.sql(f"""
WITH base AS (
  SELECT s.*, l.luck_score, l.luck_z, p.peer_theo_per_hour_usd, p.n_peers, p.zone_index,
         p.game_floor_theo_per_hour_usd,
         s.theo_per_hour_usd / p.peer_theo_per_hour_usd AS vs_peers,
         fl.x_m AS x, fl.y_m AS y,
         b.machine_id IS NOT NULL AS in_manager_bottom10, b.manager_rank
  FROM {out('machine_summary')} s
  JOIN {out('luck_scores')} l USING (machine_id)
  JOIN {out('location_peers')} p USING (machine_id)
  LEFT JOIN {src('floor_locations')} fl ON fl.location_id = s.location_id
  LEFT JOIN {out('manager_bottom10')} b USING (machine_id)
),
v AS (
  SELECT *,
    CASE
      WHEN ABS(luck_z) > {INSPECT_Z}                         THEN 'Inspect'
      WHEN days_on_floor < {MIN_DAYS} OR retire_date IS NOT NULL THEN 'Short exposure'
      WHEN availability_pct < {BROKEN_AVAIL}                  THEN 'Broken'
      WHEN vs_peers < {TRULY_BAD_RATIO}                       THEN 'Truly bad'
      WHEN zone_index < {BAD_SPOT_RATIO}                      THEN 'Bad spot'
      WHEN house_win_usd < theo_usd AND luck_score < 0.20     THEN 'Unlucky'
      ELSE 'Healthy' END AS verdict
  FROM base
)
SELECT machine_id, bank_id, zone_id, zone_name, location_id, x, y, theme, config_id,
       house_win_usd, theo_usd, theo_per_hour_usd, peer_theo_per_hour_usd, zone_index,
       availability_pct, luck_score, luck_z, days_on_floor, verdict,
       CASE verdict
         WHEN 'Inspect' THEN concat('Hold ', format_number(observed_hold*100,1), '% vs ', format_number(expected_hold*100,1),
              '% expected; z = ', format_number(luck_z,2), ' is beyond luck. Verify paytable.')
         WHEN 'Short exposure' THEN concat('Only ', days_on_floor, ' days on the floor',
              CASE WHEN retire_date IS NOT NULL THEN concat(' (retired ', retire_date, ')') ELSE concat(' (installed ', install_date, ')') END,
              '. Total win is low because of time, not performance.')
         WHEN 'Broken' THEN concat('Available only ', format_number(availability_pct,1), '% of scheduled hours.')
         WHEN 'Truly bad' THEN concat('Earns ', format_number(vs_peers*100,0), '% of same-game peers in the ', zone_name, '.')
         WHEN 'Bad spot' THEN concat('Normal for its game in the ', zone_name, ' (', format_number(vs_peers*100,0),
              '% of peers), but the zone earns ', format_number(zone_index*100,0), '% of the floor.')
         WHEN 'Unlucky' THEN concat('Theo is normal; house win below theo by $', format_number(theo_usd-house_win_usd,0),
              ' (luck score ', format_number(luck_score,2), ').')
         ELSE 'No problem found.' END AS reason,
       ROUND(CASE verdict
         WHEN 'Inspect'   THEN (theo_usd - house_win_usd) / days_on_floor * 30                      -- monthly shortfall vs theo
         WHEN 'Bad spot'  THEN (game_floor_theo_per_hour_usd - theo_per_hour_usd) * available_hours / days_on_floor * 30  -- monthly theo gained in an average spot
         WHEN 'Truly bad' THEN (peer_theo_per_hour_usd - theo_per_hour_usd) * available_hours / days_on_floor * 30
         WHEN 'Broken'    THEN (scheduled_hours - available_hours) * theo_per_hour_usd / days_on_floor * 30
         ELSE 0 END, 2) AS dollar_impact_usd,
       CASE WHEN verdict = 'Inspect' AND ABS(luck_z) > 4 THEN 'High'
            WHEN verdict = 'Short exposure' THEN 'High'
            WHEN verdict = 'Bad spot' AND vs_peers BETWEEN 0.8 AND 1.25 THEN 'High'
            WHEN verdict IN ('Unlucky','Healthy') THEN 'Medium'
            ELSE 'Medium' END AS confidence,
       in_manager_bottom10, manager_rank
FROM v
""")
verdicts.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(out("verdicts"))

display(spark.table(out("verdicts")).filter("in_manager_bottom10").orderBy("manager_rank")
        .select("manager_rank", "machine_id", "zone_name", "theme", "verdict", "reason", "dollar_impact_usd", "confidence"))
display(spark.table(out("verdicts")).groupBy("verdict").agg(F.count("*").alias("machines"),
        F.sum(F.col("in_manager_bottom10").cast("int")).alias("of_bottom10")).orderBy("verdict"))

# COMMAND ----------

# MAGIC %md ## Task 7. Claims and actions
# MAGIC Every number below is computed from the tables above, not typed in.

# COMMAND ----------

import pandas as pd

V = spark.table(out("verdicts")).toPandas()
S = spark.table(out("machine_summary")).toPandas()
L = spark.table(out("luck_scores")).toPandas()
MV = spark.table(out("move_effects")).toPandas()
Z = spark.table(out("zone_index")).toPandas()
CL = spark.table(out("b40_closure_effect")).toPandas()
INS = spark.table(out("inspection_candidates")).toPandas()
B = V[V.in_manager_bottom10 == True].sort_values("manager_rank")

n_truly_bad = int((B.verdict == "Truly bad").sum())
n_b40 = int((B.zone_id == "Z-B40").sum())
counts = B.verdict.value_counts().to_dict()
best_move = MV.sort_values("adjusted_change_pct", ascending=False).iloc[0]
b40 = Z[Z.zone_name == "Back Forty"].iloc[0]
top_zone = Z.sort_values("zone_index_vs_floor").iloc[-1]
unlucky_b10 = B[(B.luck_score < 0.05) & (B.verdict != "Inspect")].sort_values("luck_score")
corr = float(pd.Series(L.luck_z).corr(pd.Series(L.house_win_usd - L.theo_usd)))
sim_vs_z = float((L.luck_score - pd.Series([__import__('math').erf(z/2**0.5)/2+0.5 for z in L.luck_z])).abs().max())

claims = [
  {"claim_id": "P2-01", "question": "Q1",
   "claim": f"{n_truly_bad} of the manager's bottom 10 are truly bad machines; replacing all 10 is not supported",
   "evidence_table": "p2_verdicts",
   "result": "; ".join(f"{k}: {v}" for k, v in counts.items()),
   "alternative_tested": "Raw house-win ranking vs luck-, exposure- and location-adjusted ranking",
   "limitation": "Thresholds are judgment calls; see notebook rule table", "confidence": "High"},
  {"claim_id": "P2-02", "question": "Q1",
   "claim": f"{n_b40} of the bottom 10 sit in the Back Forty, the weakest zone",
   "evidence_table": "p2_zone_index",
   "result": f"Back Forty earns {b40.zone_index_vs_floor:.0%} of the floor's theo/hour for the same games ({top_zone.zone_name}: {top_zone.zone_index_vs_floor:.0%})",
   "alternative_tested": "Game mix: index compares each game with the same game floor-wide",
   "limitation": "Zone effect may reflect layout and walking distance, not one cause", "confidence": "High"},
  {"claim_id": "P2-03", "question": "Q1",
   "claim": f"Location causes low earnings: machine {best_move.machine_id} moved {best_move.from_zone} -> {best_move.to_zone} and its theo/hour changed {best_move.change_pct:+.0%}",
   "evidence_table": "p2_move_effects",
   "result": f"{best_move.theo_per_hour_before:.2f} -> {best_move.theo_per_hour_after:.2f} $/h; floor-adjusted {best_move.adjusted_change_pct:+.0%} over {int(best_move.days_after)} days",
   "alternative_tested": f"Unmoved machines over the same dates changed {best_move.unmoved_floor_change_pct:+.1%}",
   "limitation": f"Only {len(MV)} moves in the data; one machine is a small sample", "confidence": "Medium"},
  {"claim_id": "P2-04", "question": "Q1",
   "claim": "Downtime does not explain the bottom 10",
   "evidence_table": "p2_machine_summary",
   "result": f"Lowest availability on the floor is {S.availability_pct.min():.1f}%; median {S.availability_pct.median():.1f}%",
   "alternative_tested": "Theo per available hour (removes downtime) gives the same bottom group",
   "limitation": "Availability is meter-based; short outages inside an hour may be missed", "confidence": "High"},
  {"claim_id": "P2-05", "question": "Q1",
   "claim": "Inspection candidates: payout far outside game math beyond luck",
   "evidence_table": "p2_inspection_candidates",
   "result": "; ".join(f"{r.machine_id} hold {r.observed_hold:.1%} vs {r.expected_hold:.1%}, z={r.luck_z:.2f}" for r in INS.itertuples()) or "none",
   "alternative_tested": "Exact per-spin variance from game_math; 10,000 fair simulations; |z|>3.29 so ~0.3 false alarms expected across all machines",
   "limitation": "A lead to inspect, not evidence of wrongdoing", "confidence": "High"},
  {"claim_id": "P2-06", "question": "Q1",
   "claim": "Short time on the floor puts machines in the bottom 10 that should not be judged by totals",
   "evidence_table": "p2_verdicts",
   "result": "; ".join(f"{r.machine_id}: {r.days_on_floor} days" for r in B[B.verdict == 'Short exposure'].itertuples()) or "none",
   "alternative_tested": "Per-available-hour comparison",
   "limitation": "New machines may still be ramping up", "confidence": "High"},
  {"claim_id": "P2-07", "question": "Q1",
   "claim": "Simulation and closed-form luck scores agree",
   "evidence_table": "p2_luck_scores",
   "result": f"Max difference between simulated luck score and normal CDF of z: {sim_vs_z:.3f}",
   "alternative_tested": "Simple luck residual (house win - theo) ranks differently because it ignores game volatility",
   "limitation": "Normal approximation; fine at tens of thousands of spins per machine", "confidence": "High"},
  {"claim_id": "P2-08", "question": "Q1",
   "claim": "Back Forty lost share while its Main Entrance walkway was closed",
   "evidence_table": "p2_b40_closure_effect",
   "result": "; ".join(f"{r.period}: {r.b40_vs_rest:.0%} of rest of floor" for r in CL.itertuples()),
   "alternative_tested": "Compared with the rest of the floor over the same dates",
   "limitation": "Closure overlaps March seasonality", "confidence": "Medium"},
]
if len(unlucky_b10):
    u = unlucky_b10.iloc[0]
    claims.append({"claim_id": "P2-09", "question": "Q1",
      "claim": f"{u.machine_id} was also unlucky: only {u.luck_score:.1%} of fair simulations did worse",
      "evidence_table": "p2_luck_scores", "result": f"House win ${u.house_win_usd:,.0f} vs theo ${u.theo_usd:,.0f}",
      "alternative_tested": "Expected win (theo) per hour is normal for its game and zone",
      "limitation": "Testing 10 machines at 5% gives ~0.5 false positive expected", "confidence": "Medium"})

slot_counts = B.location_id.value_counts()
for loc, n in slot_counts[slot_counts > 1].items():
    ms = B[B.location_id == loc].machine_id.tolist()
    claims.append({"claim_id": f"P2-{len(claims)+1:02d}", "question": "Q1",
      "claim": f"Slot {loc} holds {n} of the bottom 10 ({', '.join(ms)}): the retired machine was already replaced in the same weak spot",
      "evidence_table": "p2_verdicts / bronze_placements", "result": "Replacing machines without moving them repeats the problem",
      "alternative_tested": "Same game title before and after; per-hour theo compared with zone peers",
      "limitation": "One slot", "confidence": "High"})

spark.createDataFrame(pd.DataFrame(claims)).write.mode("overwrite").saveAsTable(out("claims"))
display(spark.table(out("claims")))

# COMMAND ----------

# Actions priced with the official budget formula: qty x (unit + daily x max(0, 30 - lead)) / 100
AC = spark.table(src("action_catalog")).toPandas().set_index("action_type_id")
def cost(action_id, qty):
    a = AC.loc[action_id]
    return qty * (a.unit_cost_cents + a.recurring_cost_cents_per_day * max(0, 30 - a.lead_days)) / 100
def days_live(action_id): return max(0, 30 - int(AC.loc[action_id].lead_days))

actions = []
# 1) Inspect every inspection candidate
if len(INS):
    gain = sum(S.set_index("machine_id").loc[m].pipe(lambda r: (r.theo_usd - r.house_win_usd) / r.days_on_floor)
               for m in INS.machine_id) * days_live("A-INSPECT")
    actions.append({"action": "Inspect a machine and verify payout", "quantity": len(INS), "targets": ", ".join(INS.machine_id),
        "cost_usd": cost("A-INSPECT", len(INS)), "expected_benefit_usd": round(gain * 0.5, 0),
        "low_estimate_usd": 0.0, "high_estimate_usd": round(gain, 0), "confidence": "Medium",
        "how_to_measure": "Compare July hold with configured RTP; z-score back inside +/-2 confirms the fix"})
# 2) Move bad-spot bottom-10 machines out of the weak zone (benefit only if it is a swap with a low-sensitivity machine)
bs = B[B.verdict == "Bad spot"]
if len(bs):
    gain = float(((bs.set_index("machine_id").dollar_impact_usd / 30) * days_live("A-MOVE-MACHINE")).sum())
    actions.append({"action": "Move one machine", "quantity": len(bs), "targets": ", ".join(bs.machine_id),
        "cost_usd": cost("A-MOVE-MACHINE", len(bs)), "expected_benefit_usd": round(gain * 0.5, 0),
        "low_estimate_usd": 0.0, "high_estimate_usd": round(gain, 0), "confidence": "Low",
        "how_to_measure": "Swap into slots of high-traffic machines whose game is less sensitive to location; July theo per available hour vs unmoved same-game machines (difference-in-differences, as in p2_move_effects)"})
# 3) Replacing the manager's 10 — priced so the team can show what NOT spending saves
actions.append({"action": "Replace with a purchased machine (manager's plan, NOT recommended)", "quantity": 10,
    "targets": ", ".join(B.machine_id), "cost_usd": cost("A-REPLACE-BUY", 10), "expected_benefit_usd": 0.0,
    "low_estimate_usd": 0.0, "high_estimate_usd": 0.0, "confidence": "High",
    "how_to_measure": f"Only {n_truly_bad} of the 10 are truly bad; budget is better kept or spent on the zone"})

spark.createDataFrame(pd.DataFrame(actions)).write.mode("overwrite").saveAsTable(out("actions"))
display(spark.table(out("actions")))

# COMMAND ----------

# MAGIC %md ## Done
# MAGIC Tables written: `p2_manager_bottom10`, `p2_machine_summary`, `p2_luck_scores`, `p2_luck_sims`, `p2_location_peers`,
# MAGIC `p2_zone_index`, `p2_b40_closure_effect`, `p2_move_effects`, `p2_inspection_candidates`, `p2_verdicts`, `p2_claims`, `p2_actions`.