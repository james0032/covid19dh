"""
Combined temporal analysis: LR probe vs Transformer
- 7-day, monthly, and quarterly sliding windows
- Per-scale plots + all-scales combined plots
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "covid19dh_usa_party_joint.csv")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")

FEATURES = [
    "school_closing", "workplace_closing", "cancel_events",
    "gatherings_restrictions", "transport_closing", "stay_home_restrictions",
    "internal_movement_restrictions", "international_movement_restrictions",
    "information_campaigns", "testing_policy", "contact_tracing",
    "facial_coverings", "vaccination_policy", "elderly_people_protection",
]
STATE_COL = "State_x"

# Color scheme: same model = same family; darker = coarser time scale (quarterly darkest)
COLORS = {
    # LR probe — blue family
    "lr_quarterly": "#1a3a6b",   # darkest
    "lr_monthly":   "#2e6db4",
    "lr_7d":        "#89b4e8",   # lightest
    # Transformer — orange family
    "tf_quarterly": "#7a2800",   # darkest
    "tf_monthly":   "#c94a00",
    "tf_7d":        "#f4a46a",   # lightest
}
LINE_WIDTHS = {"quarterly": 3, "monthly": 2, "7d": 1}

# ── Load & preprocess ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.dropna(subset=FEATURES).copy()

le_party = LabelEncoder()
df["party_enc"] = le_party.fit_transform(df["Party"])

le_ppl = LabelEncoder()
df["ppl_enc"] = le_ppl.fit_transform(df["Party of Partisan Lean"])

print(f"Rows: {len(df):,}  |  States: {df[STATE_COL].nunique()}")

# ── Build per-state sequences (for Transformer) ───────────────────────────────
states     = np.array(sorted(df[STATE_COL].unique()))
date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

sequences, party_labels, lean_labels, ppl_labels = [], [], [], []
for state in states:
    sdf = df[df[STATE_COL] == state].set_index("date").reindex(date_range)
    sequences.append(sdf[FEATURES].fillna(0).values.astype(np.float32))
    party_labels.append(df[df[STATE_COL] == state]["party_enc"].iloc[0])
    lean_labels.append(df[df[STATE_COL] == state]["Partisan Lean"].iloc[0])
    ppl_labels.append(df[df[STATE_COL] == state]["ppl_enc"].iloc[0])

y_party_seq = np.array(party_labels)
state_names = states

# ── Canonical split: rng.permutation on sorted state list (matches all scripts)
n_states       = len(state_names)
rng            = np.random.RandomState(42)
perm           = rng.permutation(n_states)
split          = int(0.7 * n_states)
tr_idx         = perm[:split]
va_idx         = perm[split:]
train_states   = set(state_names[tr_idx])
test_states    = set(state_names[va_idx])
val_state_list = list(state_names[va_idx])

print(f"Train: {len(tr_idx)} states  |  Val: {len(va_idx)} states")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transformer model ─────────────────────────────────────────────────────────
class PartyTransformer(nn.Module):
    def __init__(self, n_features=14, d_model=32, nhead=4, num_layers=2, n_out=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,
                                               dim_feedforward=128, dropout=dropout)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier  = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, n_out))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(-1)

model_party = PartyTransformer(n_features=len(FEATURES), n_out=1).to(device)
model_party.load_state_dict(torch.load(os.path.join(OUT_DIR, "transformer_party_best.pt"), map_location=device))
model_party.eval()

model_reg = PartyTransformer(n_features=len(FEATURES), n_out=1).to(device)
model_reg.load_state_dict(torch.load(os.path.join(OUT_DIR, "transformer_lean_reg_best.pt"), map_location=device))
model_reg.eval()

print("Transformer checkpoints loaded.")

probe_party = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
probe_ppl   = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
probe_reg   = LinearRegression()

# ── Helper: build val-state sequences for a date window ──────────────────────
def build_val_seqs(window_dates):
    seqs, y_p, y_l, y_ppl = [], [], [], []
    for state in val_state_list:
        sdf = df[df[STATE_COL] == state].set_index("date").reindex(window_dates)
        seqs.append(sdf[FEATURES].fillna(0).values.astype(np.float32))
        row = df[df[STATE_COL] == state].iloc[0]
        y_p.append(row["party_enc"])
        y_l.append(row["Partisan Lean"])
        y_ppl.append(row["ppl_enc"])
    return np.stack(seqs), np.array(y_p), np.array(y_l, dtype=np.float32), np.array(y_ppl)

def score_window(window_dates, tr_wdf, te_wdf):
    """Compute LR probe + Transformer scores for one window."""
    rec = {}
    if len(tr_wdf["party_enc"].unique()) >= 2:
        probe_party.fit(tr_wdf[FEATURES], tr_wdf["party_enc"])
        rec["lr_party_f1"] = f1_score(te_wdf["party_enc"],
                                       probe_party.predict(te_wdf[FEATURES]),
                                       average="weighted", zero_division=0)
    if len(tr_wdf["ppl_enc"].unique()) >= 2:
        probe_ppl.fit(tr_wdf[FEATURES], tr_wdf["ppl_enc"])
        rec["lr_ppl_f1"] = f1_score(te_wdf["ppl_enc"],
                                     probe_ppl.predict(te_wdf[FEATURES]),
                                     average="weighted", zero_division=0)
    if len(tr_wdf) > 0:
        probe_reg.fit(tr_wdf[FEATURES], tr_wdf["Partisan Lean"])
        rec["lr_lean_r2"] = r2_score(te_wdf["Partisan Lean"],
                                      probe_reg.predict(te_wdf[FEATURES]))
    seqs, y_p, y_l, y_ppl = build_val_seqs(window_dates)
    X_te = torch.tensor(seqs).to(device)
    with torch.no_grad():
        preds_party = (model_party(X_te).cpu().numpy() > 0).astype(int)
        rec["tf_party_f1"] = f1_score(y_p, preds_party, average="weighted", zero_division=0)
        rec["tf_ppl_f1"]   = f1_score(y_ppl, preds_party, average="weighted", zero_division=0)
        rec["tf_lean_r2"]  = r2_score(y_l, model_reg(X_te).cpu().numpy())
    return rec

# ── Quarterly window loop (anchored to 2020-03-01, 91-day windows) ────────────
print("\nRunning quarterly windows...")
anchor   = pd.Timestamp("2020-03-01")
q_starts = pd.date_range(anchor, df["date"].max(), freq="91D")
quarterly = []

for q_start in q_starts:
    q_end   = q_start + pd.Timedelta(days=90)
    q_dates = pd.date_range(q_start, q_end, freq="D")
    qdf     = df[df["date"].isin(q_dates)].dropna(subset=FEATURES)
    tr_qdf  = qdf[qdf[STATE_COL].isin(train_states)]
    te_qdf  = qdf[qdf[STATE_COL].isin(test_states)]
    if len(te_qdf) == 0:
        continue
    rec = {"date": q_start.strftime("%Y-%m-%d"), "date_dt": q_start.to_pydatetime(),
           **score_window(q_dates, tr_qdf, te_qdf)}
    quarterly.append(rec)

quarterly_df = pd.DataFrame(quarterly)
print(f"  Quarterly windows: {len(quarterly_df)}")

# ── Monthly window loop ───────────────────────────────────────────────────────
print("Running monthly windows...")
months  = pd.period_range(df["date"].min(), df["date"].max(), freq="M")
monthly = []

for month in months:
    mdf    = df[df["date"].dt.to_period("M") == month].dropna(subset=FEATURES)
    tr_mdf = mdf[mdf[STATE_COL].isin(train_states)]
    te_mdf = mdf[mdf[STATE_COL].isin(test_states)]
    if len(te_mdf) == 0:
        continue
    m_dates = pd.date_range(str(month.start_time.date()), str(month.end_time.date()), freq="D")
    rec = {"date": str(month), "date_dt": month.start_time.to_pydatetime(),
           **score_window(m_dates, tr_mdf, te_mdf)}
    monthly.append(rec)

monthly_df = pd.DataFrame(monthly)
print(f"  Monthly windows: {len(monthly_df)}")

# ── 7-day window loop ─────────────────────────────────────────────────────────
print("Running 7-day windows...")
all_dates  = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
records_7d = []

for i in range(len(all_dates) - 7 + 1):
    window_dates = all_dates[i : i + 7]
    wdf    = df[df["date"].isin(window_dates)].dropna(subset=FEATURES)
    tr_wdf = wdf[wdf[STATE_COL].isin(train_states)]
    te_wdf = wdf[wdf[STATE_COL].isin(test_states)]
    if len(te_wdf) == 0:
        continue
    rec = {"date": str(window_dates[0].date()), "date_dt": window_dates[0].to_pydatetime(),
           **score_window(window_dates, tr_wdf, te_wdf)}
    records_7d.append(rec)

df_7d = pd.DataFrame(records_7d)
print(f"  7-day windows: {len(df_7d)}")

# ── Per-model combined plots (all three time scales per model) ────────────────
fig_lr = go.Figure()
fig_tf = go.Figure()

for tdf, scale, scale_key in [
    (quarterly_df, "Quarterly", "quarterly"),
    (monthly_df,   "Monthly",   "monthly"),
    (df_7d,        "7-day",     "7d"),
]:
    w = LINE_WIDTHS[scale_key]
    if "lr_party_f1" in tdf.columns:
        fig_lr.add_trace(go.Scatter(
            x=tdf["date_dt"], y=tdf["lr_party_f1"], mode="lines", name=scale,
            customdata=tdf["date"],
            hovertemplate="%{customdata}: %{y:.3f}<extra>" + scale + "</extra>",
            line=dict(color=COLORS[f"lr_{scale_key}"], width=w),
        ))
    if "tf_party_f1" in tdf.columns:
        fig_tf.add_trace(go.Scatter(
            x=tdf["date_dt"], y=tdf["tf_party_f1"], mode="lines", name=scale,
            customdata=tdf["date"],
            hovertemplate="%{customdata}: %{y:.3f}<extra>" + scale + "</extra>",
            line=dict(color=COLORS[f"tf_{scale_key}"], width=w),
        ))

fig_lr.update_layout(
    title="Party Predictability — LR Probe (All Time Scales)",
    xaxis_title="Date", yaxis_title="Weighted F1",
    yaxis=dict(range=[0, 1]), hovermode="x unified",
)
fig_tf.update_layout(
    title="Party Predictability — Transformer (All Time Scales)",
    xaxis_title="Date", yaxis_title="Weighted F1",
    yaxis=dict(range=[0, 1]), hovermode="x unified",
)

fig_lr.write_html(os.path.join(OUT_DIR, "temporal_party_f1_lr.html"))
fig_tf.write_html(os.path.join(OUT_DIR, "temporal_party_f1_tf.html"))
print("  Saved temporal_party_f1_lr.html, temporal_party_f1_tf.html")

# ── Per-model PPL F1 plots ────────────────────────────────────────────────────
fig_lr_ppl = go.Figure()
fig_tf_ppl = go.Figure()

for tdf, scale, scale_key in [
    (quarterly_df, "Quarterly", "quarterly"),
    (monthly_df,   "Monthly",   "monthly"),
    (df_7d,        "7-day",     "7d"),
]:
    w = LINE_WIDTHS[scale_key]
    if "lr_ppl_f1" in tdf.columns:
        fig_lr_ppl.add_trace(go.Scatter(
            x=tdf["date_dt"], y=tdf["lr_ppl_f1"], mode="lines", name=scale,
            customdata=tdf["date"],
            hovertemplate="%{customdata}: %{y:.3f}<extra>" + scale + "</extra>",
            line=dict(color=COLORS[f"lr_{scale_key}"], width=w),
        ))
    if "tf_ppl_f1" in tdf.columns:
        fig_tf_ppl.add_trace(go.Scatter(
            x=tdf["date_dt"], y=tdf["tf_ppl_f1"], mode="lines", name=scale,
            customdata=tdf["date"],
            hovertemplate="%{customdata}: %{y:.3f}<extra>" + scale + "</extra>",
            line=dict(color=COLORS[f"tf_{scale_key}"], width=w),
        ))

fig_lr_ppl.update_layout(
    title="Party of Partisan Lean Predictability — LR Probe (All Time Scales)",
    xaxis_title="Date", yaxis_title="Weighted F1",
    yaxis=dict(range=[0, 1]), hovermode="x unified",
)
fig_tf_ppl.update_layout(
    title="Party of Partisan Lean Predictability — Transformer (All Time Scales)",
    xaxis_title="Date", yaxis_title="Weighted F1",
    yaxis=dict(range=[0, 1]), hovermode="x unified",
)

fig_lr_ppl.write_html(os.path.join(OUT_DIR, "temporal_ppl_f1_lr.html"))
fig_tf_ppl.write_html(os.path.join(OUT_DIR, "temporal_ppl_f1_tf.html"))
print("  Saved temporal_ppl_f1_lr.html, temporal_ppl_f1_tf.html")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\nTop-5 — Party F1 (LR probe, monthly):")
print(monthly_df.nlargest(5, "lr_party_f1")[["date", "lr_party_f1", "tf_party_f1"]].to_string(index=False))
print("\nTop-5 — Party F1 (Transformer, monthly):")
print(monthly_df.nlargest(5, "tf_party_f1")[["date", "lr_party_f1", "tf_party_f1"]].to_string(index=False))
print("\nQuarterly summary:")
print(quarterly_df[["date", "lr_party_f1", "tf_party_f1", "lr_lean_r2", "tf_lean_r2"]].to_string(index=False))

fig_lr.show()
fig_tf.show()
fig_lr_ppl.show()
fig_tf_ppl.show()
