# Team Rules

1. Read from `workspace.casino.bronze_*` tables. Never change them.
2. Save your results as `workspace.casino.p1_*`, `p2_*`, `p3_*` (Person 4 uses `mock_*` for fake data).
3. Money rules: always dollars (`_usd`), group by `gaming_date`, use ratio of sums (SUM/SUM), never average of ratios.
4. Drop rows where `location_id` is empty (machine not on the floor).
5. Save findings to your own claims table: `pX_claims`
   Columns: claim_id, question, claim, evidence_table, result, alternative_tested, limitation, confidence
6. Save suggested actions to your own actions table: `pX_actions`
   Columns: action, quantity, targets, cost_usd, expected_benefit_usd, low_estimate_usd, high_estimate_usd, confidence, how_to_measure
7. Never rename or remove columns from the agreed table contracts. Adding columns is fine.
8. Only edit files in your own folder. Person 4 owns `app/`.
9. Never commit data files or the Databricks token.

## Who does what
| Person | Folder | Questions | Saves |
|---|---|---|---|
| P1 | p1_downtime_pipeline/ | Q3 downtime, Q5 comps, pipeline | p1_downtime, p1_comps, p1_leak_clues, p1_claims, p1_actions |
| P2 | p2_machine_verdicts/ | Q1 machine verdicts | p2_verdicts, p2_luck_sims, p2_move_effects, p2_claims, p2_actions |
| P3 | p3_demand_forecast/ | Q2 demand, Q4 players, forecast model | p3_demand, p3_forecast_results, p3_model_metrics, p3_players, p3_concentration, p3_claims, p3_actions |
| P4 | app/ | Dashboard, Genie, submission | reads all tables |
