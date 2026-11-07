# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · P1 Lakeflow pipeline: Bronze → Silver → Gold
# MAGIC **Run this as a Lakeflow Declarative Pipeline (not as a normal notebook).**
# MAGIC Settings: catalog `workspace`, schema `casino`, serverless, triggered mode.
# MAGIC
# MAGIC | Layer | Tables | What happens |
# MAGIC |---|---|---|
# MAGIC | Bronze | `bronze_*` (made by notebook 00) | raw copy, never edited |
# MAGIC | Silver | `silver_machine_day`, `silver_operations`, `silver_zone_traffic_hourly`, `silver_comp_redemptions` | cleaning rules as **expectations**: drop off-floor days, `scheduled_seconds` for day length, cents → dollars, offline counters = unknown (not 0) |
# MAGIC | Gold | `p1_downtime`, `p1_downtime_events`, `p1_comps`, `p1_comp_rate_daily`, `p1_leak_clues` | the tables the dashboard reads |
# MAGIC
# MAGIC Every rule lives in one dictionary, so the same rule is (a) a Databricks expectation that shows up in the
# MAGIC pipeline's data-quality tab and (b) a column that sends failing rows to `p1_leak_clues`.

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from pyspark.sql import Window as W

SRC = spark.conf.get("p1.source_schema", "workspace.casino")

# The raw parquet timestamps load as TIMESTAMP_NTZ; Delta needs that feature switched on per table.
_dlt_table = dlt.table
def _table_with_ntz(*args, **kwargs):
    props = dict(kwargs.pop("table_properties", None) or {})
    props.setdefault("delta.feature.timestampNtz", "supported")
    return _dlt_table(*args, table_properties=props, **kwargs)
dlt.table = _table_with_ntz


def bronze(name):
    return spark.read.table(f"{SRC}.bronze_{name}")


def bronze_exists(name):
    try:
        return spark.catalog.tableExists(f"{SRC}.bronze_{name}")
    except Exception:
        return False


# Shifts follow the slot-tech roster: day 06-14, swing 14-22, grave 22-06 (local clock)
def shift_from_local_hour(col):
    h = F.hour(col)
    return (F.when((h >= 6) & (h < 14), "day")
             .when((h >= 14) & (h < 22), "swing")
             .otherwise("grave"))


def gaming_date_from_local(col):
    # gaming day runs 06:00 -> 06:00, so shift the clock back 6 hours and take the date
    return F.to_date(col - F.expr("INTERVAL 6 HOURS"))

# COMMAND ----------

# MAGIC %md ## Rules (expectations). If a row fails one of these it becomes a leak clue.

# COMMAND ----------

MACHINE_DAY_RULES = {
    "money_adds_up":          "house_win_cents = coin_in_cents - payout_cents",
    "available_le_scheduled": "available_seconds <= scheduled_seconds",
    "occupied_le_available":  "occupied_seconds <= available_seconds",
    "no_play_while_down":     "NOT (available_seconds = 0 AND paid_spins > 0)",
    "carded_le_total":        "carded_coin_in_cents <= coin_in_cents",
    "hits_le_spins":          "payout_positive_spins <= paid_spins",
    "theo_matches_rtp":       "abs(theo_cents - coin_in_cents * (1 - configured_rtp)) <= 100",
    "avg_bet_in_range":       "paid_spins = 0 OR (coin_in_cents / paid_spins BETWEEN min_bet_cents AND max_bet_cents)",
    "payout_in_increments":   "payout_cents % payout_increment_cents = 0",
}

OPERATION_RULES = {
    "work_after_open":     "work_started_at_utc >= opened_at_utc",
    "ticket_after_open":   "recorded_at_utc >= opened_at_utc",
    "outage_under_24h":    "down_hours <= 24",
}

# COMMAND ----------

# MAGIC %md ## Silver

# COMMAND ----------

@dlt.table(name="silver_calendar_day", comment="Gaming days with real length (scheduled_seconds; 23h on 2025-03-08).")
@dlt.expect_or_fail("valid_day_length", "scheduled_seconds IN (82800, 86400, 90000)")
def silver_calendar_day():
    return (bronze("calendar_day")
            .withColumn("scheduled_hours", F.col("scheduled_seconds") / 3600.0))


@dlt.table(name="silver_machine_day",
           comment="One row per machine x gaming day ON THE FLOOR. Dollars, hours, placement and game math joined.")
@dlt.expect_or_drop("on_floor", "location_id IS NOT NULL")        # drops the 1,018 not-installed / retired days
@dlt.expect_all(MACHINE_DAY_RULES)                                 # warn + count; failures go to p1_leak_clues
def silver_machine_day():
    md = bronze("machine_day")
    gm = bronze("game_math").select("config_id", "configured_rtp", "payout_multiplier_variance",
                                    "min_bet_cents", "max_bet_cents", "payout_increment_cents", "volatility_class")
    pdv = bronze("prepared_daily_view").select("machine_id", "gaming_date", "zone_id", "zone_name",
                                               "bank_id", "manufacturer", "lease_class")
    df = (md.join(gm, "config_id", "left")
            .join(pdv, ["machine_id", "gaming_date"], "left")
            .withColumn("month", F.date_format("gaming_date", "yyyy-MM"))
            # hours: use the machine's own scheduled_seconds (handles the 23-hour DST day)
            .withColumn("scheduled_hours", F.col("scheduled_seconds") / 3600.0)
            .withColumn("available_hours", F.col("available_seconds") / 3600.0)
            .withColumn("occupied_hours", F.col("occupied_seconds") / 3600.0)
            .withColumn("downtime_hours", (F.col("scheduled_seconds") - F.col("available_seconds")) / 3600.0)
            .withColumn("availability_pct",
                        F.when(F.col("scheduled_seconds") > 0,
                               100.0 * F.col("available_seconds") / F.col("scheduled_seconds")))
            # cents -> dollars (only here, once)
            .withColumn("coin_in_usd", F.col("coin_in_cents") / 100.0)
            .withColumn("payout_usd", F.col("payout_cents") / 100.0)
            .withColumn("house_win_usd", F.col("house_win_cents") / 100.0)
            .withColumn("theo_usd", (F.col("theo_cents") / 100.0).cast("double"))
            .withColumn("carded_coin_in_usd", F.col("carded_coin_in_cents") / 100.0))
    # same rules, as a column: list of rule names this row fails
    failed = [F.when(~F.expr(cond), F.lit(name)) for name, cond in MACHINE_DAY_RULES.items()]
    return df.withColumn("rules_failed", F.filter(F.array(*failed), lambda x: x.isNotNull()))


@dlt.table(name="silver_operations",
           comment="Every operations ticket with manufacturer, shift, wait/fix/down hours and slot-tech staffing.")
@dlt.expect_or_drop("valid_interval", "available_at_utc >= opened_at_utc")
@dlt.expect_all(OPERATION_RULES)
def silver_operations():
    ops = (bronze("operations")
           .join(bronze("machines").select("machine_id", "manufacturer"), "machine_id", "left")
           .withColumn("gaming_date", gaming_date_from_local(F.col("opened_at_local")))
           .withColumn("month", F.date_format("gaming_date", "yyyy-MM"))
           .withColumn("down_hours", (F.unix_timestamp("available_at_utc") - F.unix_timestamp("opened_at_utc")) / 3600.0)
           .withColumn("wait_hours", (F.unix_timestamp("work_started_at_utc") - F.unix_timestamp("opened_at_utc")) / 3600.0)
           .withColumn("fix_hours", (F.unix_timestamp("available_at_utc") - F.unix_timestamp("work_started_at_utc")) / 3600.0)
           .withColumn("direct_cost_usd", F.col("direct_cost_cents") / 100.0))

    if bronze_exists("shift_roster"):
        roster = (bronze("shift_roster").where("role = 'slot_tech'")
                  .select(F.col("shift"), F.col("start_utc"), F.col("end_utc"),
                          F.col("planned_headcount").alias("techs_planned"),
                          F.col("actual_headcount").alias("techs_on_shift")))
        ops = (ops.alias("o")
               .join(roster.alias("r"),
                     (F.col("o.opened_at_utc") >= F.col("r.start_utc")) & (F.col("o.opened_at_utc") < F.col("r.end_utc")),
                     "left")
               .drop("start_utc", "end_utc")
               .withColumn("shift", F.coalesce(F.col("shift"), shift_from_local_hour(F.col("opened_at_local")))))
    else:
        ops = (ops.withColumn("shift", shift_from_local_hour(F.col("opened_at_local")))
                  .withColumn("techs_planned", F.lit(None).cast("int"))
                  .withColumn("techs_on_shift", F.lit(None).cast("int")))

    ops = ops.withColumn("short_staffed", F.col("techs_on_shift") < F.col("techs_planned"))
    # placement at the time of the ticket (bank / zone)
    pdv = bronze("prepared_daily_view").select("machine_id", "gaming_date", "zone_id", "bank_id")
    return ops.join(pdv, ["machine_id", "gaming_date"], "left")


@dlt.table(name="silver_zone_traffic_hourly",
           comment="Zone counters. Offline hours keep entries = NULL (unknown), never 0.")
@dlt.expect("counter_online", "counter_status = 'ok'")
def silver_zone_traffic_hourly():
    return (bronze("zone_traffic_hourly")
            .withColumn("gaming_date", gaming_date_from_local(F.col("hour_start_local")))
            .withColumn("is_observed", F.col("counter_status") != "offline")
            .withColumn("entries_count",
                        F.when(F.col("counter_status") == "offline", F.lit(None)).otherwise(F.col("entries_count"))))


@dlt.table(name="silver_comp_redemptions",
           comment="Loyalty redemptions (comps). economic_cost is the house's cost and is ALREADY inside POS cogs.")
@dlt.expect("cost_le_face", "economic_cost_cents <= face_value_cents")
@dlt.expect("points_spent", "points_delta < 0")
def silver_comp_redemptions():
    return (bronze("loyalty_ledger").where("event_type = 'redeem'")
            .withColumn("event_at_local", F.from_utc_timestamp("event_at_utc", "America/Los_Angeles"))
            .withColumn("gaming_date", gaming_date_from_local(F.col("event_at_local")))
            .withColumn("month", F.date_format("gaming_date", "yyyy-MM"))
            .withColumn("points_spent", -F.col("points_delta"))
            .withColumn("face_value_usd", F.col("face_value_cents") / 100.0)
            .withColumn("comp_cost_usd", F.col("economic_cost_cents") / 100.0))

# COMMAND ----------

# MAGIC %md ## Gold

# COMMAND ----------

def _theo_rate(md):
    """Theo per AVAILABLE hour: machine-month, falling back to machine overall, then zone-month."""
    mm = (md.where("available_hours > 0").groupBy("machine_id", "month")
            .agg((F.sum("theo_usd") / F.sum("available_hours")).alias("r_mm")))
    m = (md.where("available_hours > 0").groupBy("machine_id")
           .agg((F.sum("theo_usd") / F.sum("available_hours")).alias("r_m")))
    zm = (md.where("available_hours > 0").groupBy("zone_id", "month")
            .agg((F.sum("theo_usd") / F.sum("available_hours")).alias("r_zm")))
    return mm, m, zm


@dlt.table(name="p1_downtime",
           comment="CONTRACT. Machine x gaming day: availability and theo lost to downtime, with cause.")
def p1_downtime():
    md = dlt.read("silver_machine_day")
    cal = dlt.read("silver_calendar_day").select("gaming_date", "gaming_day_start_utc", "gaming_day_end_utc")
    ops = dlt.read("silver_operations").select("machine_id", "event_type", "opened_at_utc", "available_at_utc")

    # which ticket types overlap each machine's gaming day (an outage can cross the 06:00 boundary)
    causes = (ops.join(cal, (ops.opened_at_utc < cal.gaming_day_end_utc) & (ops.available_at_utc > cal.gaming_day_start_utc))
                 .groupBy("machine_id", "gaming_date")
                 .agg(F.array_join(F.array_sort(F.collect_set("event_type")), "+").alias("ops_cause")))

    mm, m, zm = _theo_rate(md)
    return (md.join(causes, ["machine_id", "gaming_date"], "left")
              .join(mm, ["machine_id", "month"], "left").join(m, "machine_id", "left").join(zm, ["zone_id", "month"], "left")
              .withColumn("theo_per_available_hour", F.coalesce("r_mm", "r_m", "r_zm", F.lit(0.0)))
              .withColumn("lost_theo_usd", F.round(F.col("downtime_hours") * F.col("theo_per_available_hour"), 2))
              .withColumn("cause", F.when(F.col("downtime_hours") <= 0, "none")
                                     .otherwise(F.coalesce("ops_cause", F.lit("unexplained"))))
              .select("machine_id", "gaming_date", "bank_id", "zone_id", "manufacturer",
                      F.round("available_hours", 4).alias("available_hours"),
                      F.round("scheduled_hours", 4).alias("scheduled_hours"),
                      F.round("availability_pct", 3).alias("availability_pct"),
                      "lost_theo_usd", "cause"))


@dlt.table(name="p1_downtime_events",
           comment="One row per operations ticket: shift, staffing, wait vs repair time, theo lost.")
def p1_downtime_events():
    ops = dlt.read("silver_operations")
    mm, m, _ = _theo_rate(dlt.read("silver_machine_day"))
    return (ops.join(mm, ["machine_id", "month"], "left").join(m, "machine_id", "left")
               .withColumn("theo_per_available_hour", F.coalesce("r_mm", "r_m", F.lit(0.0)))
               .withColumn("lost_theo_usd", F.round(F.col("down_hours") * F.col("theo_per_available_hour"), 2))
               .select("operation_id", "machine_id", "manufacturer", "zone_id", "bank_id", "event_type",
                       "opened_at_local", "gaming_date", "month", "shift", "techs_planned", "techs_on_shift",
                       "short_staffed", "wait_hours", "fix_hours", "down_hours", "theo_per_available_hour",
                       "lost_theo_usd", "direct_cost_usd"))


@dlt.table(name="p1_comps", comment="CONTRACT. Comp cost and points by month and tier (tracked players only).")
def p1_comps():
    # comp_cost_usd is already inside linked_pos_cogs_usd: we report it, we never subtract both
    return (bronze("player_period")
            .groupBy(F.col("period_month").alias("month"), F.col("tier_end").alias("tier"))
            .agg(F.round(F.sum("comp_cost_usd"), 2).cast("double").alias("comp_cost_usd"),
                 F.sum("points_earned").alias("points_earned"),
                 F.sum("points_redeemed").alias("points_redeemed"))
            .select("month", "comp_cost_usd", "points_earned", "points_redeemed", "tier"))


@dlt.table(name="p1_comp_rate_daily",
           comment="Daily redemptions: how many points one comp dollar costs a player (shows the 2025-05-01 change).")
def p1_comp_rate_daily():
    r = dlt.read("silver_comp_redemptions")
    return (r.groupBy("gaming_date", "month", F.col("tier_before").alias("tier"))
             .agg(F.count("*").alias("redemptions"),
                  F.sum("points_spent").alias("points_redeemed"),
                  F.round(F.sum("face_value_usd"), 2).alias("face_value_usd"),
                  F.round(F.sum("comp_cost_usd"), 2).alias("comp_cost_usd"))
             .withColumn("points_per_comp_dollar", F.round(F.col("points_redeemed") / F.col("face_value_usd"), 3)))


@dlt.table(name="p1_leak_clues",
           comment="CONTRACT. Rows worth a second look. A clue is a lead to inspect, never an accusation.")
def p1_leak_clues():
    md = dlt.read("silver_machine_day")
    ops = dlt.read("silver_operations")

    # 1) any accounting rule failed on a machine-day
    rule_fail = (md.where(F.size("rules_failed") > 0)
                   .select("machine_id", "gaming_date", F.explode("rules_failed").alias("rule_failed"),
                           F.format_string("coin_in=$%.2f payout=$%.2f house_win=$%.2f avail_h=%.2f spins=%d",
                                           "coin_in_usd", "payout_usd", "house_win_usd", "available_hours",
                                           "paid_spins").alias("details")))

    # 2) machine available almost all day but nobody played it -> silent fault or dead spot?
    dead = (md.where("paid_spins = 0 AND available_hours >= 20")
              .select("machine_id", "gaming_date", F.lit("zero_play_while_available").alias("rule_failed"),
                      F.format_string("available %.1f h, 0 spins, zone %s, bank %s",
                                      "available_hours", "zone_id", "bank_id").alias("details")))

    # 3) very long outage
    longo = (ops.where("down_hours > 12")
                .select("machine_id", "gaming_date", F.lit("outage_over_12h").alias("rule_failed"),
                        F.format_string("%s: down %.1f h (waited %.1f h for a tech), shift %s",
                                        "event_type", "down_hours", "wait_hours", "shift").alias("details")))

    # 4) machine broke again within 72 h of coming back: the repair may not be holding
    f = ops.where("event_type = 'fault_repair'")
    w = W.partitionBy("machine_id").orderBy("opened_at_utc")
    rep = (f.withColumn("prev_back", F.lag("available_at_utc").over(w))
             .where(F.col("opened_at_utc") < F.col("prev_back") + F.expr("INTERVAL 72 HOURS"))
             .select("machine_id", "gaming_date", F.lit("repeat_fault_within_72h").alias("rule_failed"),
                     F.format_string("failed %.1f h after the previous repair",
                                     (F.unix_timestamp("opened_at_utc") - F.unix_timestamp("prev_back")) / 3600.0)
                      .alias("details")))

    # 5) period hold far from configured RTP after allowing for luck.
    #    SD of house win for a machine = sqrt(sum(bet^2) * payout variance per unit bet). |z| > 3 = inspect.
    z = (md.where("paid_spins > 0").groupBy("machine_id")
           .agg(F.max("gaming_date").alias("gaming_date"),
                F.sum("house_win_usd").alias("hw"), F.sum("theo_usd").alias("theo"),
                F.sum("coin_in_usd").alias("ci"),
                (F.sqrt(F.sum(F.col("sum_bet_squared") * F.col("payout_multiplier_variance"))) / 100.0).alias("sd"))
           .withColumn("z", (F.col("hw") - F.col("theo")) / F.col("sd"))
           .where(F.abs("z") > 3)
           .select("machine_id", "gaming_date", F.lit("hold_far_from_rtp_period").alias("rule_failed"),
                   F.format_string("Jan-Jun hold %.2f%% vs expected %.2f%%; house win $%.0f vs theo $%.0f; z=%.2f "
                                   "(normal approx, heavy-tailed games) -> inspect, not an accusation",
                                   100 * F.col("hw") / F.col("ci"), 100 * F.col("theo") / F.col("ci"),
                                   "hw", "theo", "z").alias("details")))

    return rule_fail.unionByName(dead).unionByName(longo).unionByName(rep).unionByName(z)