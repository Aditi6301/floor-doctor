# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # P3 · 03 Player outcomes (Q4)
# MAGIC **Question:** What did tracked (loyalty-card) players wager, get back and net, and how concentrated are those results?
# MAGIC
# MAGIC - Player tables only see **carded** play, so totals will not match machine meters. That's expected.
# MAGIC - Report honestly. **No ideas that push heavy losers to gamble more**; protective observations are welcome.
# MAGIC - Sign convention used here: `net_usd` = what players got back minus what they wagered (**negative = players lost**).

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_col(cols, must_have, prefer=None, avoid=None):
    hits = [c for c in cols if all(k in c.lower() for k in must_have)]
    if avoid:
        hits = [c for c in hits if not any(a in c.lower() for a in avoid)]
    if prefer:
        pref = [c for c in hits if prefer in c.lower()]
        if pref:
            return pref[0]
    return hits[0] if hits else None

def to_usd(s, col):
    return s / 100 if "cents" in col.lower() else s

pp = spark.table("workspace.casino.bronze_player_period").toPandas()
pl = spark.table("workspace.casino.bronze_players").toPandas()
print("player_period columns:", list(pp.columns))
print("players columns:", list(pl.columns))

# COMMAND ----------

ID = find_col(pp.columns, ["player"], prefer="id")
PERIOD = find_col(pp.columns, ["month"]) or find_col(pp.columns, ["period"])
WAGER = find_col(pp.columns, ["coin_in"], prefer="usd") or find_col(pp.columns, ["wager"], prefer="usd")
WONBACK = find_col(pp.columns, ["payout"], prefer="usd") or find_col(pp.columns, ["won"], prefer="usd")
NET = find_col(pp.columns, ["net"], prefer="usd")
THEO = find_col(pp.columns, ["theo"], prefer="usd") or find_col(pp.columns, ["expected"], prefer="usd")
print("Guesses -> id:", ID, "| period:", PERIOD, "| wager:", WAGER, "| won back:", WONBACK, "| net:", NET, "| theo:", THEO)

PL_ID = find_col(pl.columns, ["player"], prefer="id")
TIER = find_col(pl.columns, ["tier"])
AGE = find_col(pl.columns, ["age"])
REGION = find_col(pl.columns, ["region"]) or find_col(pl.columns, ["home"])
print("Guesses -> players id:", PL_ID, "| tier:", TIER, "| age:", AGE, "| region:", REGION)

# Overrides (fill if a guess is wrong):
# ID, PERIOD, WAGER, WONBACK, NET, THEO = "player_id", "period_month", "coin_in_usd", "payout_usd", "net_win_usd", "theo_usd"
# PL_ID, TIER, AGE, REGION = "player_id", "starting_tier", "age_band", "home_region"

# COMMAND ----------

d = pd.DataFrame({
    "player_id": pp[ID],
    "month": pd.to_datetime(pp[PERIOD].astype(str).str[:10], errors="coerce").dt.to_period("M").astype(str),
    "wagered_usd": to_usd(pp[WAGER], WAGER),
})
if WONBACK:
    d["won_back_usd"] = to_usd(pp[WONBACK], WONBACK)
    d["net_usd"] = d["won_back_usd"] - d["wagered_usd"]
else:
    # Only a net column exists. Check its sign: if most players show positive net, it's the house's win.
    raw_net = to_usd(pp[NET], NET)
    house_view = (raw_net > 0).mean() > 0.5
    print("Net column looks like", "HOUSE win (flipping sign)" if house_view else "PLAYER result")
    d["net_usd"] = -raw_net if house_view else raw_net
    d["won_back_usd"] = d["wagered_usd"] + d["net_usd"]
if THEO:
    d["expected_loss_usd"] = to_usd(pp[THEO], THEO)

meta = pl[[PL_ID] + [c for c in [TIER, AGE, REGION] if c]].rename(
    columns={PL_ID: "player_id", TIER: "tier", AGE: "age_band", REGION: "home_region"})
for c in ["tier", "age_band", "home_region"]:
    if c not in meta.columns:
        meta[c] = "unknown"
d = d.merge(meta, on="player_id", how="left").fillna({"tier": "unknown", "age_band": "unknown", "home_region": "unknown"})
display(d.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Totals

# COMMAND ----------

tot_w, tot_b, tot_n = d["wagered_usd"].sum(), d["won_back_usd"].sum(), d["net_usd"].sum()
n_players = d["player_id"].nunique()
print(f"Carded players:   {n_players:,}")
print(f"Total wagered:    ${tot_w:,.0f}")
print(f"Total won back:   ${tot_b:,.0f}")
print(f"Players' net:     ${tot_n:,.0f}  (negative = players lost)")
print(f"Effective hold on carded play: {-tot_n / tot_w:.2%}")
if "expected_loss_usd" in d:
    print(f"Expected loss (theo): ${d['expected_loss_usd'].sum():,.0f}  vs actual loss ${-tot_n:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Concentration: how much of the total loss comes from the biggest losers?

# COMMAND ----------

per_player = d.groupby("player_id", as_index=False).agg(
    wagered_usd=("wagered_usd", "sum"), net_usd=("net_usd", "sum"))
per_player["loss_usd"] = -per_player["net_usd"]
total_loss = per_player["loss_usd"].sum()
ranked = per_player.sort_values("loss_usd", ascending=False).reset_index(drop=True)
ranked["cum_share"] = ranked["loss_usd"].cumsum() / total_loss

rows = []
for p in [0.1, 1, 5, 10, 20, 50, 100]:
    k = max(1, int(round(len(ranked) * p / 100)))
    rows.append({"top_pct": float(p), "share_of_net_loss": float(ranked["loss_usd"].iloc[:k].sum() / total_loss)})
conc = pd.DataFrame(rows)
display(conc.assign(share_pct=lambda x: (x.share_of_net_loss * 100).round(1)))

winners = (per_player["net_usd"] > 0).mean()
print(f"Share of carded players who came out ahead: {winners:.1%}")

# COMMAND ----------

# Lorenz-style curve: players sorted from biggest loser, cumulative share of total loss
x = np.arange(1, len(ranked) + 1) / len(ranked) * 100
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, ranked["cum_share"] * 100, label="Cumulative share of total loss")
ax.plot([0, 100], [0, 100], "--", label="Equal split")
ax.set_xlabel("% of carded players (biggest losers first)")
ax.set_ylabel("% of total net loss")
ax.set_title("Concentration of player losses")
ax.legend(); plt.show()
# Note: the curve can go above 100% before coming back down, because winners offset losses.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. By guest group (judges will drill into these)

# COMMAND ----------

for g in ["tier", "age_band", "home_region"]:
    grp = d.groupby(g).agg(players=("player_id", "nunique"), wagered_usd=("wagered_usd", "sum"),
                           net_usd=("net_usd", "sum"))
    grp["loss_per_player_usd"] = -grp["net_usd"] / grp["players"]
    grp["share_of_loss_pct"] = grp["net_usd"] / tot_n * 100
    print(f"\n--- by {g} ---")
    display(grp.round(1).reset_index().sort_values("share_of_loss_pct", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. By month

# COMMAND ----------

by_month = d.groupby("month").agg(players=("player_id", "nunique"), wagered_usd=("wagered_usd", "sum"),
                                  net_usd=("net_usd", "sum"))
by_month["hold_pct"] = -by_month["net_usd"] / by_month["wagered_usd"] * 100
display(by_month.round(1).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Save tables

# COMMAND ----------

players_out = d.groupby(["month", "tier", "age_band", "home_region"], as_index=False).agg(
    wagered_usd=("wagered_usd", "sum"), won_back_usd=("won_back_usd", "sum"),
    net_usd=("net_usd", "sum"), player_count=("player_id", "nunique"))
for c in ["month", "tier", "age_band", "home_region"]:
    players_out[c] = players_out[c].astype(str)
spark.createDataFrame(players_out).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("workspace.casino.p3_players")
spark.createDataFrame(conc).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("workspace.casino.p3_concentration")
print("Saved p3_players and p3_concentration")
# Note: player_count per group/month counts distinct players in that row; don't sum it across months.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Notes for claims

# COMMAND ----------

top1 = conc.loc[conc.top_pct == 1, "share_of_net_loss"].iloc[0]
top10 = conc.loc[conc.top_pct == 10, "share_of_net_loss"].iloc[0]
print(f"Top 1% of carded players account for {top1:.0%} of total net loss; top 10% for {top10:.0%}.")
print(f"{winners:.0%} of carded players came out ahead over the six months.")