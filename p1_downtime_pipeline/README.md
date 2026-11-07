# Person 1: Pipeline, Downtime (Q3) and Comps (Q5)

Everything here was tested end to end against the real `dashboard-royale` parquet files (local Spark), so the
numbers below are what you should see in Databricks.

## Files

| File | Type | Run how |
|---|---|---|
| `00_load_bronze_tables.py` | notebook | Run once by hand. Creates every `bronze_*` table and checks row counts. |
| `01_p1_pipeline.py` | **Lakeflow pipeline source** | Do not "Run all". Attach it to a pipeline (step 3). |
| `02_p1_analysis_claims_actions.py` | notebook | Run after the pipeline. Makes summaries, claims and priced actions. |

## Steps (about 30 minutes)

**1. Upload the missing file.** Catalog → `workspace.casino.raw` → Upload: `data/5_optional/shift_roster.parquet`
(needed for the short-staffed shift finding). Also upload `4_context/real_sources/*` for P3 while you are there.

**2. Grants + Bronze.** In the SQL Editor:
```sql
GRANT USE CATALOG ON CATALOG workspace TO `account users`;
GRANT ALL PRIVILEGES ON SCHEMA workspace.casino TO `account users`;
```
Then open `00_load_bronze_tables.py` → Run all. Every line should say `OK`. Tell the team Bronze is ready.

**3. Create the pipeline.** Left bar → Jobs & Pipelines → Create → ETL pipeline
- Name: `floor_doctor_p1`
- Source code: `p1_downtime_pipeline/01_p1_pipeline.py` from your Git folder
- Default catalog `workspace`, default schema `casino`
- Compute: Serverless · Pipeline mode: Triggered
- Start. First run takes about 5 minutes. Open the graph view: you should see 5 silver and 5 gold tables.

> If someone (P4's mock data) already created a table called `p1_downtime`, `p1_comps` or `p1_leak_clues` in
> `workspace.casino`, drop it first. A pipeline can't take over a table it didn't create.

**4. Run `02_p1_analysis_claims_actions.py`** (Run all). It saves `p1_downtime_summary`, `p1_comp_change`,
`p1_claims`, `p1_actions`.

**5. Job (automatic refresh).** Jobs & Pipelines → Create → Job `floor_doctor_refresh`:
- Task 1 `pipeline`: type *Pipeline* → `floor_doctor_p1`
- Task 2 `analysis`: type *Notebook* → `02_p1_analysis_claims_actions.py`, depends on Task 1
- (Later P2/P3 add their notebooks as Tasks 3/4 depending on Task 1)
- Schedule: daily 06:30 America/Chicago (just after the gaming day closes). Run it once now.

**6. Screenshots for the video / write-up** (save in `p1_downtime_pipeline/screenshots/`):
1. Pipeline graph (Bronze → Silver → Gold)
2. Pipeline → click `silver_machine_day` → **Data quality** tab (shows 1,018 rows dropped by `on_floor`, 0 failures on 9 money rules)
3. Catalog → `p1_downtime` → **Lineage** tab → *See lineage graph* (expand upstream to `bronze_*`)
4. Job run page showing both tasks green

## What it produces

| Table | Contract? | Rows | Notes |
|---|---|---:|---|
| `p1_downtime` | yes | 55,092 | `availability_pct` is 0–100. `cause` = ticket types overlapping that gaming day, or `none` |
| `p1_comps` | yes | 18 | month × tier (tier at month end). Sum across tiers for floor totals |
| `p1_leak_clues` | yes | 165 | rules: `outage_over_12h`, `repeat_fault_within_72h`, `zero_play_while_available`, `hold_far_from_rtp_period`, plus any accounting-rule failure |
| `p1_claims` | yes | 11 | |
| `p1_actions` | yes | 4 | |
| `p1_downtime_events` | extra | 1,226 | one row per ticket: shift, techs planned/on shift, wait vs fix hours |
| `p1_downtime_summary` | extra | 21 | by manufacturer / shift / month / staffing / zone (page 4) |
| `p1_comp_rate_daily` | extra | 542 | points per comp dollar by day and tier (page 7 headline chart) |
| `p1_comp_change` | extra | 6 | before vs after 2025-05-01 with 95% intervals |

## Headline findings (from the real data)

**Q3 Downtime**
- Machines were available **99.62%** of scheduled hours: 5,021 machine-hours down, about **$90.7K theo lost** (0.39% of theo).
- Every downtime hour matches a ticket. Faults = 98% of downtime; **28% of fault downtime is waiting for a tech**, not repairing.
- **Ridgeline fails 3.6× more often than Halcyon** (4.04 vs 1.11 faults per 100 machine-days).
- **Grave shift waits 1.96 h** for a tech vs 0.85–0.90 h on other shifts. Repair time is the same on every shift, so this is a staffing problem.
- Short-staffed shifts: wait **2.25 h vs 0.66 h** (+1.6 h, 95% CI 1.1–2.1). Worst month: **May** (avg wait 2.2 h).
- **Money takeaway:** an extra grave tech recovers about $610 of theo in 30 days for $11,520. Preventive maintenance on Ridgeline roughly breaks even on theo, against a $9,000 cost. **Downtime is real, but too cheap to fix with this budget.** That is an honest "keep the money" argument.

**Q5 Comps**
- Comps cost **$190.7K** (Jan–Jun). That cost is already inside the $937.9K POS cost of goods, so it is counted once.
- **On 2025-05-01 the point price of comps was cut overnight from 13.1 to 5.8 points per comp dollar (−56%).**
- Comp cost per $1,000 carded theo went **+9.5%** ($11.95 → $13.09, CI +0.65 to +1.67), but **carded play did not rise** (−1.6%, CI includes 0). The change cost money and bought no measurable play.
- Platinum comp cost per month tripled ($2.6K → $8.4K) as players reach Platinum: tier creep is a second driver.
- Actions: reverting floor-wide saves about $2.6K, but with a wide downside. The **comp pilot ($4,000)** is the safer buy because it answers the question with a control group.

**Pipeline / trust:** all 55,092 on-floor machine-days pass 9 accounting rules. Low earners are **not** explained by meter errors. One machine (M-1158, hold 3.0% vs expected 10%, z = −4.6) is flagged for P2's luck test and inspection.

## Sync with the team
- **P2:** compare your availability with `p1_downtime.availability_pct` at hour 19. Also pass `M-1158` (from `p1_leak_clues`) to your inspect list.
- **P4:** page 4 = `p1_downtime_summary` + `p1_downtime_events`; page 7 = `p1_comp_rate_daily` (the May 1 drop is the chart) + `p1_comp_change`.
