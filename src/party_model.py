import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
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

# ── Load & preprocess ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.dropna(subset=FEATURES).copy()
le = LabelEncoder()
df["party_enc"] = le.fit_transform(df["Party"])  # Democratic=0, Republican=1
print(f"Usable rows after dropping NaN: {len(df):,}")
print(f"Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── Part 1: Static models ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART 1 — Static Models (all dates pooled)")
print("="*60)

X = df[FEATURES].values
y = df["party_enc"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results = {}
importances = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1  = f1_score(y_test, pred, average="weighted")
    results[name] = {"accuracy": acc, "f1": f1}
    print(f"\n── {name} ──")
    print(f"  Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
    print(classification_report(y_test, pred, target_names=le.classes_))

    if hasattr(model, "coef_"):
        importances[name] = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importances[name] = model.feature_importances_

# Feature importance plot
fig_imp = make_subplots(rows=1, cols=3, subplot_titles=list(importances.keys()))
for col, (name, imp) in enumerate(importances.items(), 1):
    order = np.argsort(imp)
    fig_imp.add_trace(
        go.Bar(x=imp[order], y=[FEATURES[i] for i in order], orientation="h", name=name),
        row=1, col=col,
    )
fig_imp.update_layout(title="Feature Importance / Coefficients per Model", height=500, showlegend=False)
imp_path = os.path.join(OUT_DIR, "party_feature_importance.html")
fig_imp.write_html(imp_path)
print(f"\nFeature importance plot saved to: {os.path.normpath(imp_path)}")

# ── Part 2: Transformer Classifier ───────────────────────────────────────────
print("\n" + "="*60)
print("PART 2 — Transformer Classifier (per-state time series)")
print("="*60)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Build per-state sequences — align on common date range
states = df["State"].unique()
date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
T = len(date_range)
date_index = {d: i for i, d in enumerate(date_range)}

sequences, labels_seq, state_names = [], [], []
for state in states:
    sdf = df[df["State"] == state].set_index("date").reindex(date_range)
    seq = sdf[FEATURES].fillna(0).values.astype(np.float32)  # (T, 14)
    party = df[df["State"] == state]["party_enc"].iloc[0]
    sequences.append(seq)
    labels_seq.append(party)
    state_names.append(state)

X_seq = np.stack(sequences)          # (n_states, T, 14)
y_seq = np.array(labels_seq)

# Train/val split by state
n = len(states)
idx = np.random.RandomState(42).permutation(n)
split = int(0.8 * n)
train_idx, val_idx = idx[:split], idx[split:]

X_tr = torch.tensor(X_seq[train_idx])
y_tr = torch.tensor(y_seq[train_idx], dtype=torch.float32)
X_val = torch.tensor(X_seq[val_idx])
y_val = torch.tensor(y_seq[val_idx], dtype=torch.float32)

class PartyTransformer(nn.Module):
    def __init__(self, n_features=14, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=64)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):              # x: (B, T, F)
        x = self.input_proj(x)        # (B, T, d_model)
        x = self.transformer(x)       # (B, T, d_model)
        x = x.mean(dim=1)             # (B, d_model)  mean-pool over time
        return self.classifier(x).squeeze(-1)  # (B,)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_t = PartyTransformer(n_features=len(FEATURES)).to(device)
optimizer = torch.optim.Adam(model_t.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=8, shuffle=True)
for epoch in range(30):
    model_t.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model_t(xb), yb)
        loss.backward()
        optimizer.step()

model_t.eval()
with torch.no_grad():
    logits = model_t(X_val.to(device)).cpu().numpy()
    pred_t = (logits > 0).astype(int)
acc_t = accuracy_score(y_val.numpy(), pred_t)
f1_t  = f1_score(y_val.numpy(), pred_t, average="weighted")
print(f"Transformer — Val Accuracy: {acc_t:.4f}  |  Val F1: {f1_t:.4f}")
print(classification_report(y_val.numpy(), pred_t, target_names=le.classes_))

# ── Part 3: Temporal accuracy (sliding monthly window) ────────────────────────
print("\n" + "="*60)
print("PART 3 — Temporal Accuracy (sliding monthly window)")
print("="*60)

all_states = df["State"].unique()
n_states = len(all_states)
state_party = df.drop_duplicates("State").set_index("State")["party_enc"].to_dict()

# Hold out 20% of states consistently
rng = np.random.RandomState(42)
perm = rng.permutation(n_states)
split = int(0.8 * n_states)
train_states = set(all_states[perm[:split]])
test_states  = set(all_states[perm[split:]])

months = pd.period_range(df["date"].min(), df["date"].max(), freq="M")
probe = LogisticRegression(max_iter=1000, random_state=42)

records = []
for month in months:
    mdf = df[(df["date"].dt.to_period("M") == month)]
    mdf = mdf.dropna(subset=FEATURES)
    if len(mdf) == 0:
        continue

    tr = mdf[mdf["State"].isin(train_states)]
    te = mdf[mdf["State"].isin(test_states)]
    if len(tr["party_enc"].unique()) < 2 or len(te) == 0:
        continue

    probe.fit(tr[FEATURES], tr["party_enc"])
    pred_m = probe.predict(te[FEATURES])
    acc_m = accuracy_score(te["party_enc"], pred_m)
    f1_m  = f1_score(te["party_enc"], pred_m, average="weighted", zero_division=0)
    records.append({"month": str(month), "accuracy": acc_m, "f1": f1_m})

temp_df = pd.DataFrame(records)

fig_t = go.Figure()
fig_t.add_trace(go.Scatter(x=temp_df["month"], y=temp_df["accuracy"], mode="lines+markers", name="Accuracy"))
fig_t.add_trace(go.Scatter(x=temp_df["month"], y=temp_df["f1"], mode="lines+markers", name="F1 (weighted)"))
fig_t.update_layout(
    title="Party Predictability Over Time (Sliding Monthly Window)",
    xaxis_title="Month", yaxis_title="Score",
    yaxis=dict(range=[0, 1]),
    hovermode="x unified",
)
temp_path = os.path.join(OUT_DIR, "party_temporal_accuracy.html")
fig_t.write_html(temp_path)
print(f"Temporal accuracy plot saved to: {os.path.normpath(temp_path)}")

print("\nTop-5 months (best predictability):")
print(temp_df.nlargest(5, "accuracy")[["month", "accuracy", "f1"]].to_string(index=False))

print("\nBottom-5 months (lowest predictability):")
print(temp_df.nsmallest(5, "accuracy")[["month", "accuracy", "f1"]].to_string(index=False))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10}")
print("-" * 47)
for name, r in results.items():
    print(f"{name:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")
print(f"{'Transformer':<25} {acc_t:>10.4f} {f1_t:>10.4f}")
