import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "covid19dh_usa_party_joint.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

FEATURES = [
    "school_closing", "workplace_closing", "cancel_events",
    "gatherings_restrictions", "transport_closing", "stay_home_restrictions",
    "internal_movement_restrictions", "international_movement_restrictions",
    "information_campaigns", "testing_policy", "contact_tracing",
    "facial_coverings", "vaccination_policy", "elderly_people_protection",
]
EPOCHS = 100
PATIENCE = 15
STATE_COL = "State_x"

# ── Load & preprocess ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.dropna(subset=FEATURES).copy()

le_party = LabelEncoder()
df["party_enc"] = le_party.fit_transform(df["Party"])

print(f"Usable rows: {len(df):,}  |  States: {df[STATE_COL].nunique()}")

# ── Build per-state sequences ─────────────────────────────────────────────────
states     = np.array(sorted(df[STATE_COL].unique()))
date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
T = len(date_range)
print(f"Time steps: {T}  |  Features: {len(FEATURES)}")

sequences, party_labels, lean_labels = [], [], []
for state in states:
    sdf = df[df[STATE_COL] == state].set_index("date").reindex(date_range)
    seq = sdf[FEATURES].fillna(0).values.astype(np.float32)
    sequences.append(seq)
    party_labels.append(df[df[STATE_COL] == state]["party_enc"].iloc[0])
    lean_labels.append(df[df[STATE_COL] == state]["Partisan Lean"].iloc[0])

X_seq       = np.stack(sequences)
y_party_seq = np.array(party_labels)
y_lean_seq  = np.array(lean_labels, dtype=np.float32)
state_names = states

# Canonical split: rng.permutation on sorted state list (matches all scripts)
n_states = len(state_names)
rng      = np.random.RandomState(42)
perm     = rng.permutation(n_states)
split    = int(0.7 * n_states)
tr_idx   = perm[:split]
va_idx   = perm[split:]
print(f"Train: {len(tr_idx)} states  |  Val: {len(va_idx)} states")
print(f"Val states: {list(state_names[va_idx])}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Model definition ──────────────────────────────────────────────────────────
class PartyTransformer(nn.Module):
    def __init__(self, n_features=14, d_model=32, nhead=4, num_layers=2, n_out=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,
                                               dim_feedforward=128, dropout=dropout)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, n_out))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(-1)

def train_transformer(X_seq, y_seq, tr_idx, va_idx, n_out, loss_fn, metric_fn, metric_name,
                      higher_is_better, ckpt_name, y_dtype=torch.float32):
    X_tr = torch.tensor(X_seq[tr_idx])
    y_tr = torch.tensor(y_seq[tr_idx], dtype=y_dtype)
    X_va = torch.tensor(X_seq[va_idx])
    y_va = torch.tensor(y_seq[va_idx], dtype=y_dtype)

    model = PartyTransformer(n_features=len(FEATURES), n_out=n_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=8, shuffle=True)

    ckpt_path = os.path.join(OUT_DIR, ckpt_name)
    best_metric = -np.inf if higher_is_better else np.inf
    best_epoch, patience_counter = 0, 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_metric": [], "val_metric": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses, train_outs, train_labels = [], [], []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward(); optimizer.step()
            train_losses.append(loss.item())
            train_outs.extend(out.detach().cpu().tolist())
            train_labels.extend(yb.cpu().tolist())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_va.to(device))
            val_loss = loss_fn(val_out, y_va.to(device)).item()
        train_metric = metric_fn(train_labels, train_outs)
        val_metric   = metric_fn(y_va.tolist(), val_out.cpu().tolist())

        history["epoch"].append(epoch)
        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(val_loss)
        history["train_metric"].append(train_metric)
        history["val_metric"].append(val_metric)

        improved = val_metric > best_metric if higher_is_better else val_metric < best_metric
        if improved:
            best_metric = val_metric; best_epoch = epoch; patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            marker = " ← best" if epoch == best_epoch else ""
            print(f"  Epoch {epoch:3d}  loss={np.mean(train_losses):.4f}  val_loss={val_loss:.4f}"
                  f"  train_{metric_name}={train_metric:.4f}  val_{metric_name}={val_metric:.4f}{marker}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best {metric_name}={best_metric:.4f} @ ep {best_epoch})")
            break

    print(f"  → Best checkpoint: epoch {best_epoch}, {metric_name}={best_metric:.4f}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, history, best_epoch, best_metric

# ── Metric helpers ────────────────────────────────────────────────────────────
def binary_f1(y_true, y_logits):
    preds = (np.array(y_logits) > 0).astype(int)
    return f1_score(np.array(y_true).astype(int), preds, average="weighted", zero_division=0)

def neg_mae(y_true, y_pred):
    return -mean_absolute_error(np.array(y_true), np.array(y_pred))

# ── Train Model 1: Binary Party ───────────────────────────────────────────────
print("\n" + "="*60 + "\nMODEL 1 — Binary Party\n" + "="*60)
model_party, hist_party, ep_party, _ = train_transformer(
    X_seq, y_party_seq, tr_idx, va_idx,
    n_out=1, loss_fn=nn.BCEWithLogitsLoss(),
    metric_fn=binary_f1, metric_name="f1",
    higher_is_better=True, ckpt_name="transformer_party_best.pt",
)

# ── Train Model 2: Partisan Lean Regression ───────────────────────────────────
print("\n" + "="*60 + "\nMODEL 2 — Partisan Lean Regression\n" + "="*60)
model_reg, hist_reg, ep_reg, _ = train_transformer(
    X_seq, y_lean_seq, tr_idx, va_idx,
    n_out=1, loss_fn=nn.MSELoss(),
    metric_fn=neg_mae, metric_name="neg_mae",
    higher_is_better=True, ckpt_name="transformer_lean_reg_best.pt",
)

# ── Final evaluations ─────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nFINAL EVALUATION\n" + "="*60)

X_va_t = torch.tensor(X_seq[va_idx]).to(device)

model_party.eval()
with torch.no_grad():
    p_logits = model_party(X_va_t).cpu().numpy()
p_preds = (p_logits > 0).astype(int)
print(f"\nBinary Party — F1: {f1_score(y_party_seq[va_idx], p_preds, average='weighted'):.4f}")
print(classification_report(y_party_seq[va_idx], p_preds, target_names=le_party.classes_))
print("Per-state:")
for i, idx in enumerate(va_idx):
    true = le_party.classes_[y_party_seq[idx]]
    pred = le_party.classes_[p_preds[i]]
    print(f"  {'✓' if true==pred else '✗'} {state_names[idx]:<25} true={true:<12} pred={pred}")

model_reg.eval()
with torch.no_grad():
    r_preds = model_reg(X_va_t).cpu().numpy()
print(f"\nLean Regression — MAE: {mean_absolute_error(y_lean_seq[va_idx], r_preds):.3f}"
      f"  R²: {r2_score(y_lean_seq[va_idx], r_preds):.4f}")

# ── Learning curves ───────────────────────────────────────────────────────────
fig_lc = make_subplots(rows=2, cols=2,
                       subplot_titles=["Party — Loss", "Party — F1",
                                       "Lean Reg — Loss", "Lean Reg — neg MAE"])

# Party loss
fig_lc.add_trace(go.Scatter(x=hist_party["epoch"], y=hist_party["train_loss"],
                             name="Party train loss", line=dict(color="royalblue")), row=1, col=1)
fig_lc.add_trace(go.Scatter(x=hist_party["epoch"], y=hist_party["val_loss"],
                             name="Party val loss", line=dict(color="tomato", dash="dash")), row=1, col=1)
# Party F1
fig_lc.add_trace(go.Scatter(x=hist_party["epoch"], y=hist_party["train_metric"],
                             name="Party train F1", line=dict(color="royalblue")), row=1, col=2)
fig_lc.add_trace(go.Scatter(x=hist_party["epoch"], y=hist_party["val_metric"],
                             name="Party val F1", line=dict(color="tomato", dash="dash")), row=1, col=2)
# Lean Reg loss
fig_lc.add_trace(go.Scatter(x=hist_reg["epoch"], y=hist_reg["train_loss"],
                             name="Lean Reg train loss", line=dict(color="seagreen")), row=2, col=1)
fig_lc.add_trace(go.Scatter(x=hist_reg["epoch"], y=hist_reg["val_loss"],
                             name="Lean Reg val loss", line=dict(color="darkorange", dash="dash")), row=2, col=1)
# Lean Reg neg MAE
fig_lc.add_trace(go.Scatter(x=hist_reg["epoch"], y=hist_reg["train_metric"],
                             name="Lean Reg train neg MAE", line=dict(color="seagreen")), row=2, col=2)
fig_lc.add_trace(go.Scatter(x=hist_reg["epoch"], y=hist_reg["val_metric"],
                             name="Lean Reg val neg MAE", line=dict(color="darkorange", dash="dash")), row=2, col=2)

fig_lc.update_layout(title="Transformer Learning Curves", height=700, hovermode="x unified")
fig_lc.write_html(os.path.join(OUT_DIR, "transformer_learning_curves.html"))
print(f"\nLearning curves saved.")

# ── Temporal sliding window ───────────────────────────────────────────────────
print("\n" + "="*60 + "\nTEMPORAL ANALYSIS\n" + "="*60)

months = pd.period_range(df["date"].min(), df["date"].max(), freq="M")

for m in [model_party, model_reg]: m.eval()
temp_records = []

for month in months:
    mdf = df[df["date"].dt.to_period("M") == month].dropna(subset=FEATURES)
    if len(mdf) == 0:
        continue
    month_dates = pd.date_range(str(month.start_time.date()), str(month.end_time.date()), freq="D")

    seqs_te, y_party_te, y_lean_te = [], [], []
    for state in state_names[va_idx]:
        sdf = mdf[mdf[STATE_COL] == state].set_index("date").reindex(month_dates)
        seq = sdf[FEATURES].fillna(0).values.astype(np.float32)
        seqs_te.append(seq)
        row = df[df[STATE_COL] == state].iloc[0]
        y_party_te.append(row["party_enc"])
        y_lean_te.append(row["Partisan Lean"])

    if len(seqs_te) == 0:
        continue
    X_te_m = torch.tensor(np.stack(seqs_te)).to(device)

    with torch.no_grad():
        p_f1 = f1_score(y_party_te, (model_party(X_te_m).cpu().numpy() > 0).astype(int),
                        average="weighted", zero_division=0)
        r2_m = r2_score(y_lean_te, model_reg(X_te_m).cpu().numpy())

    temp_records.append({"month": str(month), "party_f1": p_f1, "lean_r2": r2_m})

temp_df = pd.DataFrame(temp_records)

# F1 plot — Party binary only
fig_f1 = go.Figure()
fig_f1.add_trace(go.Scatter(x=temp_df["month"], y=temp_df["party_f1"],
                             mode="lines+markers", name="Party (binary)", line=dict(color="royalblue")))
fig_f1.update_layout(title=f"Transformer Party Predictability — F1 (checkpoint: ep {ep_party})",
                     xaxis_title="Month", yaxis_title="Weighted F1",
                     yaxis=dict(range=[0, 1]), hovermode="x unified")
fig_f1.write_html(os.path.join(OUT_DIR, "transformer_temporal_f1.html"))

# R² plot — regression
fig_r2 = go.Figure()
fig_r2.add_trace(go.Scatter(x=temp_df["month"], y=temp_df["lean_r2"],
                             mode="lines+markers", name="Partisan Lean (regression)", line=dict(color="seagreen")))
fig_r2.update_layout(title=f"Transformer Lean Regression Predictability — R² (checkpoint: ep {ep_reg})",
                     xaxis_title="Month", yaxis_title="R²", hovermode="x unified")
fig_r2.write_html(os.path.join(OUT_DIR, "transformer_temporal_r2.html"))

print(f"\nTemporal F1 plot: outputs/transformer_temporal_f1.html")
print(f"Temporal R² plot: outputs/transformer_temporal_r2.html")

print("\nTop-5 months — Party F1:")
print(temp_df.nlargest(5, "party_f1")[["month", "party_f1"]].to_string(index=False))
print("\nTop-5 months — Lean R²:")
print(temp_df.nlargest(5, "lean_r2")[["month", "lean_r2"]].to_string(index=False))

fig_f1.show()
fig_r2.show()
