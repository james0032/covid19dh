# Technical Report: Transformer Model for COVID-19 Policy Partisanship

## 1. Overview

This report documents the Transformer model used to predict the **partisan affiliation** of US state governors from daily COVID-19 policy responses. Two prediction targets are modeled simultaneously:

- **Binary party classification**: Democratic (0) vs. Republican (1) governor, evaluated with weighted F1
- **Partisan lean regression**: continuous score (−49.7 to +68.2) indicating degree of partisan lean, evaluated with R²

The model leverages the full temporal trajectory of a state's policy choices — not just a single-day snapshot — making it fundamentally different from the static classifiers (Logistic Regression, Random Forest, Gradient Boosting) also evaluated in this project.

---

## 2. Dataset

**Source file**: `data/covid19dh_usa_party_joint.csv`

- **77,897** state-day rows across **50** US states and territories
- Date range: 2020-01-01 to 2022-12-31 (1,096 days)
- Joined from:
  - COVID-19 Data Hub (US state level, Level 2)
  - Governor party affiliation (`data/state_party_affiliation_2020_2021.csv`)
  - Partisan lean scores (`data/pol_lean.csv`)

**Class distribution**:
| Party | State-day rows |
|---|---|
| Democratic | 37,497 |
| Republican | 40,400 |

---

## 3. Input Features

Fourteen ordinal policy measures from the Oxford COVID-19 Government Response Tracker are used as input features. Each value is an integer on a scale specific to that measure:

| Feature | Scale | Description |
|---|---|---|
| `school_closing` | 0–3 | School closure level |
| `workplace_closing` | 0–3 | Workplace closure level |
| `cancel_events` | 0–2 | Public event cancellations |
| `gatherings_restrictions` | 0–4 | Restrictions on gatherings |
| `transport_closing` | 0–2 | Public transport closure |
| `stay_home_restrictions` | 0–3 | Stay-at-home orders |
| `internal_movement_restrictions` | 0–2 | Internal movement restrictions |
| `international_movement_restrictions` | 0–4 | International travel controls |
| `information_campaigns` | 0–2 | Public information campaigns |
| `testing_policy` | 0–3 | COVID-19 testing policy |
| `contact_tracing` | 0–2 | Contact tracing policy |
| `facial_coverings` | 0–4 | Facial covering mandates |
| `vaccination_policy` | 0–5 | Vaccination rollout policy |
| `elderly_people_protection` | 0–3 | Elderly protection measures |

---

## 4. Input Data Processing Pipeline

### 4.1 Loading and Cleaning

```python
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.dropna(subset=FEATURES).copy()
```

Rows missing any of the 14 features are dropped. The remaining rows are used for all downstream steps.

### 4.2 Label Encoding

```python
le_party = LabelEncoder()
df["party_enc"] = le_party.fit_transform(df["Party"])
# Democratic = 0, Republican = 1
```

### 4.3 Building Per-State Time Series

The core transformation converts the tabular row-per-day format into a 3D tensor of shape `(n_states, T, 14)`.

```python
date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
# → 1,096 calendar days (2020-01-01 to 2022-12-31)

for state in states:
    sdf = df[df[STATE_COL] == state].set_index("date").reindex(date_range)
    seq = sdf[FEATURES].fillna(0).values.astype(np.float32)
    sequences.append(seq)
```

Key design decisions:
- **Reindex to the global date range**: every state gets a fixed-length sequence of T=1,096 time steps, regardless of how many days of data it actually has
- **Fill missing days with 0**: days where a state has no recorded policy entry are treated as "no active policy" (all features = 0), which is a reasonable assumption for early 2020 before policies were enacted
- **dtype float32**: required for PyTorch tensor conversion

Final tensors:
```
X_seq       : shape (n_states, 1096, 14), float32
y_party_seq : shape (n_states,), int     [0 or 1]
y_lean_seq  : shape (n_states,), float32 [continuous]
```

### 4.4 Train/Validation Split by State

To prevent data leakage, the split is performed **at the state level**, not the row level. Because the same state appears on 1,096 consecutive days, a row-level split would allow the model to see policy data from a held-out state during training.

```python
rng = np.random.RandomState(42)
dem_idx = np.where(y_party_seq == 0)[0]; rng.shuffle(dem_idx)
rep_idx = np.where(y_party_seq == 1)[0]; rng.shuffle(rep_idx)

tr_idx = np.concatenate([dem_idx[:int(0.8*len(dem_idx))],
                          rep_idx[:int(0.8*len(rep_idx))]])
va_idx = np.concatenate([dem_idx[int(0.8*len(dem_idx)):],
                          rep_idx[int(0.8*len(rep_idx)):]])
```

- **Stratified by party**: 80% of Democratic states and 80% of Republican states go to training separately, ensuring the validation set contains both parties
- **RandomState(42)**: fully reproducible split
- **Result**: ~39 training states, ~10 validation states

---

## 5. Model Architecture

Both the party classification and partisan lean regression tasks use the same `PartyTransformer` architecture. They differ only in their loss function and output interpretation.

```
Input: (batch_size, T, 14)
       ↓
  Linear(14 → 32)                          ← input_proj: project features to d_model
       ↓
  TransformerEncoder(
    num_layers = 2,
    nhead      = 4,
    d_model    = 32,
    dim_feedforward = 128,
    dropout    = 0.1,
    batch_first = True
  )                                         ← self-attention over T time steps
       ↓
  mean(dim=1)                               ← global average pooling over T
       ↓
  Linear(32 → 16) → ReLU → Linear(16 → 1) ← task head
       ↓
  Output: (batch_size,)                     ← logit (party) or scalar (lean)
```

**Parameter count** (approximate): ~20K parameters — deliberately small to avoid overfitting on a dataset of only ~50 states.

**Self-attention mechanism**: each of the 2 Transformer encoder layers applies 4-head self-attention over the T time steps, allowing the model to attend to any pair of days in the sequence. This captures long-range temporal dependencies (e.g., early lockdown decisions correlating with later vaccine policies) that sliding-window or static models cannot access.

**Global average pooling** (`x.mean(dim=1)`): collapses the T-dimensional temporal representation into a single fixed-size vector per state. This pooling is **length-agnostic** — the same frozen model can process sequences of any length T without modification, which is exploited in the temporal analysis (Section 7).

---

## 6. Training Procedure

Two separate model instances are trained: one for party classification, one for partisan lean regression.

### 6.1 Shared Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 5 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| LR scheduler | CosineAnnealingLR (T_max=100) |
| Batch size | 8 states |
| Max epochs | 100 |
| Early stopping patience | 15 epochs |

### 6.2 Party Classification (Model 1)

- **Loss**: `BCEWithLogitsLoss` (binary cross-entropy with logits; numerically stable)
- **Early-stopping metric**: validation weighted F1 (higher is better)
- **Threshold**: logit > 0 → Republican (1); logit ≤ 0 → Democratic (0)
- **Checkpoint**: `outputs/transformer_party_best.pt`
- **Best result**: val F1 = 0.818 at epoch 3

### 6.3 Partisan Lean Regression (Model 2)

- **Loss**: `MSELoss`
- **Early-stopping metric**: negative MAE on validation set (higher = lower error = better)
- **Output**: raw scalar, directly interpreted as predicted partisan lean score
- **Checkpoint**: `outputs/transformer_lean_reg_best.pt`
- **Best result**: val MAE ≈ 10.0, val R² ≈ 0.299 at epoch 43

### 6.4 Training Loop

```python
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in loader:                  # batches of 8 states
        out  = model(xb)                   # forward pass
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_metric = metric_fn(y_va, model(X_va))

    if val_metric improved:
        torch.save(model.state_dict(), ckpt_path)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break  # early stopping
```

At the end of training, the best checkpoint (not the final epoch) is loaded back:
```python
model.load_state_dict(torch.load(ckpt_path, map_location=device))
```

---

## 7. Temporal Sliding Window Analysis

### 7.1 Motivation

The full-sequence model sees all 1,096 days and learns which states consistently pursued stricter or looser policies over the entire pandemic. The temporal analysis asks a finer question: **at which specific periods did COVID-19 policy best reveal a state's partisan identity?**

To answer this, frozen best-checkpoint models are evaluated on sub-sequences extracted from three time scales.

### 7.2 Shared Setup

Implemented in `src/temporal_combined.py`.

- **Same state split** as training: train_states (39) and test_states (~10) are fixed
- **Frozen Transformer**: weights loaded from checkpoints; no parameter updates per window
- **LR probe**: `LogisticRegression(C=0.1, max_iter=1000)` and `LinearRegression()` retrained from scratch on each window's training data (provides a time-varying baseline)

For each window of dates, the `score_window()` function:
1. Filters the DataFrame to rows in the window; splits by train/test states
2. Fits the LR probe on train rows; predicts on test rows
3. Builds per-state sequences for the validation states from the window's date range
4. Passes sequences through both frozen Transformer models; records F1 and R²

### 7.3 Three Time Scales

#### Quarterly (12 windows)

```python
quarters = pd.period_range(df["date"].min(), df["date"].max(), freq="Q")
# 2020Q1, 2020Q2, ..., 2022Q4

for q in quarters:
    qdf     = df[df["date"].dt.to_period("Q") == q]
    q_dates = pd.date_range(q.start_time.date(), q.end_time.date(), freq="D")
    # sequence shape fed to Transformer: (n_val_states, ~91, 14)
    date_dt = q.start_time.to_pydatetime()   # first day of quarter (x-axis)
```

Each quarter provides ~91 days of context. The LR probe fits on the aggregate rows (all state-days in the quarter for train states); the Transformer receives the per-state sequence reindexed to the quarter's date range.

#### Monthly (36 windows)

```python
months = pd.period_range(df["date"].min(), df["date"].max(), freq="M")
# 2020-01, 2020-02, ..., 2022-12

for month in months:
    m_dates = pd.date_range(month.start_time.date(), month.end_time.date(), freq="D")
    # sequence shape: (n_val_states, ~30, 14)
    date_dt = month.start_time.to_pydatetime()
```

#### 7-day rolling (~1,090 windows)

```python
all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
for i in range(len(all_dates) - 7 + 1):
    window_dates = all_dates[i : i + 7]
    # sequence shape: (n_val_states, 7, 14)
    date_dt = window_dates[0].to_pydatetime()
```

The 7-day window provides the highest temporal resolution but the least context per evaluation. Consecutive windows overlap by 6 days, producing a smooth time series.

### 7.4 Sequence Construction for Each Window

For each window, validation-state sequences are built using `build_val_seqs(window_dates)`:

```python
def build_val_seqs(window_dates):
    seqs = []
    for state in val_state_list:
        sdf = df[df[STATE_COL] == state].set_index("date").reindex(window_dates)
        seqs.append(sdf[FEATURES].fillna(0).values.astype(np.float32))
    return np.stack(seqs)
# Result: (n_val_states, len(window_dates), 14)
```

The same fill-with-zero convention is used as during training. Targets (`y_party`, `y_lean`) are the state-level constants (party never changes within a state).

---

## 8. Metric Definitions

### 8.1 Weighted F1 (Party Classification)

F1 is computed over the ~10 held-out validation states for each window:

```
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 · Precision_c · Recall_c / (Precision_c + Recall_c)

F1_weighted = Σ_c  (support_c / N) · F1_c
```

where `c` iterates over classes {Democratic, Republican}, `support_c` is the number of true instances of class `c`, and N is the total number of validation states.

Weighting by support ensures that the metric reflects actual class distribution rather than treating both classes equally regardless of sample count.

**Special cases**:
- `zero_division=0`: if the LR probe sees only one party class among training states in a narrow window, it cannot learn a meaningful decision boundary. Any subsequent prediction of the missing class produces undefined precision/recall → assigned F1 = 0.
- With only ~10 validation states, F1 values are coarse (steps of ~0.1) and highly sensitive to individual state predictions.

**Transformer F1**:
```python
preds = (model_party(X_te).cpu().numpy() > 0).astype(int)
f1    = f1_score(y_party_te, preds, average="weighted", zero_division=0)
```
Logit threshold at 0 corresponds to a probability threshold of 0.5 after sigmoid.

### 8.2 R² (Partisan Lean Regression)

R² (coefficient of determination) measures how much variance in partisan lean the model explains:

```
SS_res = Σ_i (y_i − ŷ_i)²          ← residual sum of squares
SS_tot = Σ_i (y_i − ȳ)²             ← total sum of squares

R² = 1 − SS_res / SS_tot
```

Interpretation:
| R² | Meaning |
|---|---|
| 1.0 | Perfect prediction |
| 0.0 | Equivalent to predicting the mean (ȳ) for every state |
| < 0 | Worse than predicting the mean |

**Negative R² values** are expected — and informative — in the temporal analysis:

1. **Out-of-distribution context**: the Transformer was trained on 1,096-day full sequences. A 7-day or 30-day window is a qualitatively different input: the attention layers see far fewer time steps, producing a different (often noisier) pooled representation.
2. **Low signal in short windows**: policy values in a single week are nearly constant within most states, so there is little variation for the model to exploit.
3. **Small evaluation set**: with ~10 validation states, a single misprediction has large impact on R².

The quarterly Transformer R² is typically higher than monthly or 7-day R² because 91 days of policy variation give the frozen model more meaningful temporal context.

---

## 9. Visualization

All outputs are interactive Plotly HTML files saved to `outputs/`.

### Per-Scale Plots (6 files)

| File | Content |
|---|---|
| `temporal_party_f1_quarterly.html` | Party F1: LR probe vs Transformer, quarterly |
| `temporal_party_f1.html` | Party F1: LR probe vs Transformer, monthly |
| `temporal_party_f1_7d.html` | Party F1: LR probe vs Transformer, 7-day |
| `temporal_lean_r2_quarterly.html` | Lean R²: LR probe vs Transformer, quarterly |
| `temporal_lean_r2.html` | Lean R²: LR probe vs Transformer, monthly |
| `temporal_lean_r2_7d.html` | Lean R²: LR probe vs Transformer, 7-day |

### All-Scales Combined (2 files)

| File | Content |
|---|---|
| `temporal_party_f1_all.html` | Party F1 for all 6 model×scale combinations |
| `temporal_lean_r2_all.html` | Lean R² for all 6 model×scale combinations |

### X-Axis Design

All traces use a Python `datetime` object (`date_dt`) as the x-axis coordinate, not a string label. This ensures Plotly renders a **continuous, properly scaled** time axis. String labels such as "2020Q1" or "2021-03" are instead passed via `customdata` and shown in hover tooltips:

```python
fig.add_trace(go.Scatter(
    x=tdf["date_dt"],           # datetime → proper time axis
    customdata=tdf["date"],     # friendly string → hover only
    hovertemplate="%{customdata}: %{y:.3f}"
))
```

### Color Scheme

Color encodes two dimensions simultaneously: **model family** (hue) and **time scale** (darkness). Shorter windows use darker shades, reflecting that they are noisier and represent more localized measurements.

| Trace | Color | Hex |
|---|---|---|
| LR probe — 7-day | Dark blue | `#1a3a6b` |
| LR probe — monthly | Medium blue | `#2e6db4` |
| LR probe — quarterly | Light blue | `#89b4e8` |
| Transformer — 7-day | Dark orange-red | `#7a2800` |
| Transformer — monthly | Medium orange | `#c94a00` |
| Transformer — quarterly | Light orange | `#f4a46a` |

---

## 10. Key Findings

1. **Full-sequence party classification**: the Transformer achieves val F1 = 0.818, correctly classifying 9 of ~10 held-out states using only the 1,096-day policy trajectory.

2. **Partisan lean regression**: R² ≈ 0.30 on the full sequence. The continuous lean score is harder to predict, but the model explains roughly 30% of variance using policy data alone.

3. **Temporal peak of predictability**: party F1 peaks in mid-to-late 2021 across all time scales, corresponding to the vaccination rollout period when Republican and Democratic governors diverged most sharply on vaccine mandates, mask requirements, and school reopenings.

4. **Pre-pandemic baseline**: F1 drops near chance (≈ 0.5) in January–February 2020, when all states had uniformly zero policy values and there was no partisan signal to detect.

5. **Time-scale comparison**: quarterly windows yield the most stable scores for both models. The 7-day LR probe is most volatile due to the small evaluation set and low within-week policy variance. The Transformer is more stable across time scales because its frozen weights encode long-term patterns not available in a single-week snapshot.

6. **LR probe vs. Transformer**: in monthly and quarterly evaluations, both models track each other closely in F1, suggesting that the dominant signal in any given period is captured well by a linear model. The Transformer's advantage materializes primarily at the full-sequence level, where temporal ordering and long-range dependencies matter most.

---

## 11. Reproducibility

| Component | Details |
|---|---|
| Random seed | `RandomState(42)` for all splits and shuffles |
| PyTorch seed | Not explicitly fixed; results may vary slightly across runs |
| Hardware | CPU or CUDA; checkpoints saved with `map_location=device` for portability |
| Checkpoint files | `outputs/transformer_party_best.pt`, `outputs/transformer_lean_reg_best.pt` |
| Entry points | `python src/transformer_model.py` (train), `python src/temporal_combined.py` (temporal analysis) |
