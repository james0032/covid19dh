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

# Color scheme: same model = same family; darker = shorter time scale
COLORS = {
    # LR probe — blue family
    "lr_7d":        "#1a3a6b",   # darkest
    "lr_monthly":   "#2e6db4",
    "lr_quarterly": "#89b4e8",   # lightest
    # Transformer — orange family
    "tf_7d":        "#7a2800",   # darkest
    "tf_monthly":   "#c94a00",
    "tf_quarterly": "#f4a46a",   # lightest
}

# ── Load & preprocess ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.dropna(subset=FEATURES).copy()

le_party = LabelEncoder()
df["party_enc"] = le_party.fit_transform(df["Party"])

print(f"Rows: {len(df):,}  |  States: {df[STATE_COL].nunique()}")

# ── Build per-state sequences (for Transformer) ───────────────────────────────
states     = np.array(sorted(df[STATE_COL].unique()))
date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

sequences, party_labels, lean_labels = [], [], []
for state in states:
    sdf = df[df[STATE_COL] == state].set_index("date").reindex(date_range)
    sequences.append(sdf[FEATURES].fillna(0).values.astype(np.float32))
    party_labels.append(df[df[STATE_COL] == state]["party_enc"].iloc[0])
    lean_labels.append(df[df[STATE_COL] == state]["Partisan Lean"].iloc[0])

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
probe_reg   = LinearRegression()

# ── Helper: build val-state sequences for a date window ──────────────────────
def build_val_seqs(window_dates):
    seqs, y_p, y_l = [], [], []
    for state in val_state_list:
        sdf = df[df[STATE_COL] == state].set_index("date").reindex(window_dates)
        seqs.append(sdf[FEATURES].fillna(0).values.astype(np.float32))
        row = df[df[STATE_COL] == state].iloc[0]
        y_p.append(row["party_enc"])
        y_l.append(row["Partisan Lean"])
    return np.stack(seqs), np.array(y_p), np.array(y_l, dtype=np.float32)

def score_window(window_dates, tr_wdf, te_wdf):
    """Compute LR probe + Transformer scores for one window."""
    rec = {}
    if len(tr_wdf["party_enc"].unique()) >= 2:
        probe_party.fit(tr_wdf[FEATURES], tr_wdf["party_enc"])
        rec["lr_party_f1"] = f1_score(te_wdf["party_enc"],
                                       probe_party.predict(te_wdf[FEATURES]),
                                       average="weighted", zero_division=0)
    if len(tr_wdf) > 0:
        probe_reg.fit(tr_wdf[FEATURES], tr_wdf["Partisan Lean"])
        rec["lr_lean_r2"] = r2_score(te_wdf["Partisan Lean"],
                                      probe_reg.predict(te_wdf[FEATURES]))
    seqs, y_p, y_l = build_val_seqs(window_dates)
    X_te = torch.tensor(seqs).to(device)
    with torch.no_grad():
        rec["tf_party_f1"] = f1_score(y_p,
                                       (model_party(X_te).cpu().numpy() > 0).astype(int),
                                       average="weighted", zero_division=0)
        rec["tf_lean_r2"]  = r2_score(y_l, model_reg(X_te).cpu().numpy())
    return rec

# ── Quarterly window loop ─────────────────────────────────────────────────────
print("\nRunning quarterly windows...")
quarters    = pd.period_range(df["date"].min(), df["date"].max(), freq="Q")
quarterly   = []

for q in quarters:
    qdf    = df[df["date"].dt.to_period("Q") == q].dropna(subset=FEATURES)
    tr_qdf = qdf[qdf[STATE_COL].isin(train_states)]
    te_qdf = qdf[qdf[STATE_COL].isin(test_states)]
    if len(te_qdf) == 0:
        continue
    q_dates = pd.date_range(str(q.start_time.date()), str(q.end_time.date()), freq="D")
    rec = {"date": str(q), "date_dt": q.start_time.to_pydatetime(),
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

# ── Plot helper: single time scale ───────────────────────────────────────────
def make_scale_plot(tdf, lr_col, tf_col, title, y_label, lr_color, tf_color, y_range=None):
    fig = go.Figure()
    if lr_col in tdf.columns:
        fig.add_trace(go.Scatter(x=tdf["date_dt"], y=tdf[lr_col],
                                 mode="lines", name="LR probe",
                                 customdata=tdf["date"],
                                 hovertemplate="%{customdata}: %{y:.3f}<extra>LR probe</extra>",
                                 line=dict(color=lr_color)))
    if tf_col in tdf.columns:
        fig.add_trace(go.Scatter(x=tdf["date_dt"], y=tdf[tf_col],
                                 mode="lines", name="Transformer",
                                 customdata=tdf["date"],
                                 hovertemplate="%{customdata}: %{y:.3f}<extra>Transformer</extra>",
                                 line=dict(color=tf_color)))
    layout = dict(title=title, xaxis_title="Date", yaxis_title=y_label, hovermode="x unified")
    if y_range:
        layout["yaxis"] = dict(range=y_range)
    fig.update_layout(**layout)
    return fig

# ── Save per-scale plots ──────────────────────────────────────────────────────
plots = [
    # (df, scale_label, lr_col, tf_col, lr_color, tf_color, f1_file, r2_file)
    (quarterly_df, "Quarterly", "lr_party_f1", "tf_party_f1", "lr_lean_r2", "tf_lean_r2",
     COLORS["lr_quarterly"], COLORS["tf_quarterly"],
     "temporal_party_f1_quarterly.html", "temporal_lean_r2_quarterly.html"),
    (monthly_df,   "Monthly",   "lr_party_f1", "tf_party_f1", "lr_lean_r2", "tf_lean_r2",
     COLORS["lr_monthly"],   COLORS["tf_monthly"],
     "temporal_party_f1.html",           "temporal_lean_r2.html"),
    (df_7d,        "7-day",     "lr_party_f1", "tf_party_f1", "lr_lean_r2", "tf_lean_r2",
     COLORS["lr_7d"],        COLORS["tf_7d"],
     "temporal_party_f1_7d.html",        "temporal_lean_r2_7d.html"),
]

figs_to_show = []
for tdf, scale, lr_f1, tf_f1, lr_r2, tf_r2, lr_c, tf_c, f1_file, r2_file in plots:
    fig_f1 = make_scale_plot(tdf, lr_f1, tf_f1,
                              f"Party Predictability — {scale} (LR probe vs Transformer)",
                              "Weighted F1", lr_c, tf_c, y_range=[0, 1])
    fig_r2 = make_scale_plot(tdf, lr_r2, tf_r2,
                              f"Lean R² — {scale} (LR probe vs Transformer)",
                              "R²", lr_c, tf_c)
    fig_f1.write_html(os.path.join(OUT_DIR, f1_file))
    fig_r2.write_html(os.path.join(OUT_DIR, r2_file))
    figs_to_show.extend([fig_f1, fig_r2])
    print(f"  Saved {f1_file}, {r2_file}")

# ── All-scales combined plots ─────────────────────────────────────────────────
fig_all_f1 = go.Figure()
fig_all_r2 = go.Figure()

for tdf, scale, lr_key, tf_key in [
    (quarterly_df, "Quarterly", "lr_quarterly", "tf_quarterly"),
    (monthly_df,   "Monthly",   "lr_monthly",   "tf_monthly"),
    (df_7d,        "7-day",     "lr_7d",        "tf_7d"),
]:
    if "lr_party_f1" in tdf.columns:
        fig_all_f1.add_trace(go.Scatter(x=tdf["date_dt"], y=tdf["lr_party_f1"],
                                         mode="lines", name=f"LR {scale}",
                                         customdata=tdf["date"],
                                         hovertemplate="%{customdata}: %{y:.3f}<extra>LR " + scale + "</extra>",
                                         line=dict(color=COLORS[lr_key])))
    if "tf_party_f1" in tdf.columns:
        fig_all_f1.add_trace(go.Scatter(x=tdf["date_dt"], y=tdf["tf_party_f1"],
                                         mode="lines", name=f"Transformer {scale}",
                                         customdata=tdf["date"],
                                         hovertemplate="%{customdata}: %{y:.3f}<extra>Transformer " + scale + "</extra>",
                                         line=dict(color=COLORS[tf_key])))
    if "lr_lean_r2" in tdf.columns:
        fig_all_r2.add_trace(go.Scatter(x=tdf["date_dt"], y=tdf["lr_lean_r2"],
                                         mode="lines", name=f"LR {scale}",
                                         customdata=tdf["date"],
                                         hovertemplate="%{customdata}: %{y:.3f}<extra>LR " + scale + "</extra>",
                                         line=dict(color=COLORS[lr_key])))
    if "tf_lean_r2" in tdf.columns:
        fig_all_r2.add_trace(go.Scatter(x=tdf["date_dt"], y=tdf["tf_lean_r2"],
                                         mode="lines", name=f"Transformer {scale}",
                                         customdata=tdf["date"],
                                         hovertemplate="%{customdata}: %{y:.3f}<extra>Transformer " + scale + "</extra>",
                                         line=dict(color=COLORS[tf_key])))

fig_all_f1.update_layout(title="Party Predictability — All Time Scales (LR probe vs Transformer)",
                          xaxis_title="Date", yaxis_title="Weighted F1",
                          yaxis=dict(range=[0, 1]), hovermode="x unified")
fig_all_r2.update_layout(title="Lean R² — All Time Scales (LR probe vs Transformer)",
                          xaxis_title="Date", yaxis_title="R²", hovermode="x unified")

fig_all_f1.write_html(os.path.join(OUT_DIR, "temporal_party_f1_all.html"))
fig_all_r2.write_html(os.path.join(OUT_DIR, "temporal_lean_r2_all.html"))
print("  Saved temporal_party_f1_all.html, temporal_lean_r2_all.html")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\nTop-5 — Party F1 (LR probe, monthly):")
print(monthly_df.nlargest(5, "lr_party_f1")[["date", "lr_party_f1", "tf_party_f1"]].to_string(index=False))
print("\nTop-5 — Party F1 (Transformer, monthly):")
print(monthly_df.nlargest(5, "tf_party_f1")[["date", "lr_party_f1", "tf_party_f1"]].to_string(index=False))
print("\nQuarterly summary:")
print(quarterly_df[["date", "lr_party_f1", "tf_party_f1", "lr_lean_r2", "tf_lean_r2"]].to_string(index=False))

for fig in figs_to_show + [fig_all_f1, fig_all_r2]:
    fig.show()
