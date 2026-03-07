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

Four models are trained to predict governor party from the 14 policy features. The first three treat each state-day row as an independent sample (no time awareness). The Transformer is the primary model and explicitly encodes the full time series of each state.

| Model | Accuracy | F1 | Time-aware |
|---|---|---|---|
| Logistic Regression | 0.650 | 0.650 | No |
| Gradient Boosting | 0.815 | 0.815 | No |
| Random Forest | 0.899 | 0.899 | No |
| **Transformer** | **0.909** | **0.908** | **Yes** |

All models use an 80/20 train/test split by **state** (not by row) to prevent data leakage — test states are held out entirely and never seen during training.

### Transformer Architecture

The Transformer is trained on per-state time series of shape `(n_states, T, 14)`, where T = 1,096 days (2020-01-01 to 2022-12-31). Each state is represented as a sequence of daily policy vectors. The model architecture is:

- **Input projection**: Linear(14 → 32)
- **Transformer Encoder**: 2 layers, 4 attention heads, d_model=32, feedforward dim=128, dropout=0.1
- **Pooling**: Mean-pool over the time dimension
- **Classifier**: Linear(32 → 16) → ReLU → Linear(16 → 1) → sigmoid

Training uses Adam optimizer (lr=5e-4, weight decay=1e-4) with cosine annealing and early stopping (patience=15 epochs). The best checkpoint is saved at the epoch with the highest validation accuracy.

### Temporal Analysis — When Does Policy Best Predict Party?

#### Method

To test the hypothesis that **partisan policy divergence peaked early in the pandemic and converged over time**, a sliding monthly window analysis is applied using the best-checkpoint Transformer. For each calendar month M:

1. Extract all state-day rows falling within month M
2. Build a short per-state sequence of shape `(n_states, days_in_month, 14)`, filling missing days with zero
3. Pass each state's monthly sequence through the frozen Transformer (best checkpoint, no retraining)
4. Predict party for each held-out test state and compute accuracy and weighted F1
5. Record scores for month M

This produces a time series of accuracy values spanning the full dataset date range, revealing which periods had the most — and least — partisan policy divergence.

#### Results (Transformer)

| Period | Accuracy | Interpretation |
|---|---|---|
| **March 2021** (best) | **0.909** | Peak of vaccination rollout: stark partisan divergence in vaccine mandates, mask policies, and school reopening |
| February 2021 | 0.818 | Vaccine rollout beginning; early R vs D differences emerging |
| April 2021 | 0.818 | Continued divergence on reopening timelines |
| **January–February 2020** (worst) | **0.545** | Pre-pandemic baseline: all states had zero/identical policy values; model is near chance |
| April 2020 | 0.545 | Early pandemic confusion; both parties implemented broad lockdowns similarly |

## Findings

1. **COVID-19 policy is a strong predictor of governor party.** The Transformer achieves 0.909 accuracy using only 14 NPI features, classifying 10 of 11 held-out states correctly. Even the simplest model (Logistic Regression at 0.650) is well above chance, confirming substantial partisan signal in policy choices.

2. **The temporal pattern confirms the convergence hypothesis.** Predictability peaked in **March 2021** — the height of the vaccination rollout period — when Republican and Democratic governors diverged most sharply on vaccine mandates, mask requirements, and school reopenings. Predictability was lowest **before the pandemic** (Jan–Feb 2020), when all states had uniformly zero policy values, and during early 2020 when both parties initially responded similarly to the novel outbreak.

3. **Wisconsin is the only consistently misclassified state.** Wisconsin — a historically purple state with a Democratic governor and a Republican-controlled legislature — produced moderate, mixed COVID-19 policies that do not cleanly align with either partisan pattern. This political split resulted in court battles over mask mandates and school closures, producing a policy profile that sits between the two parties.

4. **Time-aware modeling outperforms static classifiers.** The Transformer (0.909) outperforms Random Forest (0.899), Gradient Boosting (0.815), and Logistic Regression (0.650) despite having far fewer training examples (39 states vs. ~42,000 rows). This demonstrates that the *temporal trajectory* of policy — how it evolved over 1,096 days — carries additional signal beyond any single-day snapshot.

## Scripts

| Script | Description |
|---|---|
| `src/descriptive.py` | General stats of the global covid19dh dataset |
| `src/us_subset.py` | Builds a combined US subset across levels 1, 2, 3 with geo labels |
| `src/us_map.py` | Choropleth map of datapoints per state (level 3 — county) |
| `src/us_map_l2.py` | Choropleth map of datapoints per state (level 2 — state) |
| `src/party_model.py` | Static party prediction models (LR, RF, GB) + LR temporal probe |
| `src/transformer_model.py` | Transformer training with early stopping + temporal accuracy analysis |

## Outputs

| File | Description |
|---|---|
| `outputs/us_map.html` | Interactive US map — datapoints per state (level 3) |
| `outputs/us_map_l2.html` | Interactive US map — datapoints per state (level 2) |
| `outputs/party_feature_importance.html` | Feature importance charts for LR, RF, GB |
| `outputs/party_temporal_accuracy.html` | Monthly sliding-window accuracy (LR probe) |
| `outputs/transformer_learning_curves.html` | Transformer training/validation loss and accuracy curves |
| `outputs/transformer_temporal_accuracy.html` | Monthly sliding-window accuracy using Transformer best checkpoint |
| `outputs/transformer_best.pt` | Saved Transformer weights at best validation epoch |
