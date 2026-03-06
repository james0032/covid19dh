# covid19dh
CAIPH Datathon 2026

## Dataset Background

This project uses the [COVID-19 Data Hub](https://covid19datahub.io/) (`covid19dh`), a unified dataset that aggregates daily COVID-19 statistics and government policy responses across 236 countries and territories from 2020-01-01 through 2024-10-19.

The dataset provides three levels of geographic granularity:
- **Level 1** — Country (236 entities, 287,783 rows globally)
- **Level 2** — State/Province (56 US states and territories, ~86,724 rows)
- **Level 3** — County (1,917 US counties, ~3.5M rows)

Key data columns include confirmed cases, deaths, recovered, hospitalizations, vaccinations, and 14 government policy/NPI (non-pharmaceutical intervention) measures indexed by the Oxford COVID-19 Government Response Tracker.

## Party Enrichment

The US state-level (level 2) data was joined with the **party affiliation of each state's governor** at the time of data collection, producing a binary classification target:

- `Democratic` — 37,497 state-day observations
- `Republican` — 40,400 state-day observations

The enriched dataset is stored at `data/covid19dh_usa_party_joint.csv` (77,897 rows × 50 columns).

## Policy Features

The following 14 ordinal policy measures (Oxford COVID-19 Government Response Tracker scale) are used as model features:

| Feature | Description |
|---|---|
| `school_closing` | School closure level (0–3) |
| `workplace_closing` | Workplace closure level (0–3) |
| `cancel_events` | Public event cancellations (0–2) |
| `gatherings_restrictions` | Restrictions on gatherings (0–4) |
| `transport_closing` | Public transport closure (0–2) |
| `stay_home_restrictions` | Stay-at-home orders (0–3) |
| `internal_movement_restrictions` | Internal movement restrictions (0–2) |
| `international_movement_restrictions` | International travel controls (0–4) |
| `information_campaigns` | Public information campaigns (0–2) |
| `testing_policy` | COVID-19 testing policy (0–3) |
| `contact_tracing` | Contact tracing policy (0–2) |
| `facial_coverings` | Facial covering mandates (0–4) |
| `vaccination_policy` | Vaccination rollout policy (0–5) |
| `elderly_people_protection` | Elderly protection measures (0–3) |

## Modeling

Three static classifiers trained on all dates pooled, and a Transformer trained on per-state time series:

| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression | 0.650 | 0.650 |
| Gradient Boosting | 0.815 | 0.815 |
| Random Forest | **0.899** | **0.899** |
| Transformer (per-state sequence) | 0.700 | 0.640 |

### Temporal Analysis — When Does Policy Best Predict Party?

A sliding monthly window analysis trains a Logistic Regression on each calendar month separately (splitting train/test by state to avoid leakage) to test the hypothesis that **partisan policy divergence peaked early in the pandemic and converged over time**.

**Key findings:**
- **Best month: March 2021** (accuracy 0.874) — peak of vaccination rollout divergence, with stark R vs D differences in mandates and reopening
- **Worst month: February 2022** (accuracy 0.379) — Omicron wave caused both parties to drop restrictions simultaneously, making states indistinguishable by policy alone
- The pattern confirms the convergence hypothesis: predictability peaked in 2020–2021 and declined sharply in 2022

## Scripts

| Script | Description |
|---|---|
| `src/descriptive.py` | General stats of the global covid19dh dataset |
| `src/us_subset.py` | Builds a combined US subset across levels 1, 2, 3 with geo labels |
| `src/us_map.py` | Choropleth map of datapoints per state (level 3 — county) |
| `src/us_map_l2.py` | Choropleth map of datapoints per state (level 2 — state) |
| `src/party_model.py` | Party prediction models + temporal accuracy analysis |

## Outputs

| File | Description |
|---|---|
| `outputs/us_map.html` | Interactive US map — datapoints per state (level 3) |
| `outputs/us_map_l2.html` | Interactive US map — datapoints per state (level 2) |
| `outputs/party_feature_importance.html` | Feature importance charts for all 3 static models |
| `outputs/party_temporal_accuracy.html` | Monthly sliding-window accuracy over time |
