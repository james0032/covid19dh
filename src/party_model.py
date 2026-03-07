import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             mean_absolute_error, r2_score)
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

STATE_COL = "State_x"

# ── Load & preprocess ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.dropna(subset=FEATURES).copy()

le_party = LabelEncoder()
df["party_enc"] = le_party.fit_transform(df["Party"])   # Democratic=0, Republican=1

print(f"Usable rows: {len(df):,}  |  States: {df[STATE_COL].nunique()}")
print(f"Party classes: {dict(zip(le_party.classes_, le_party.transform(le_party.classes_)))}")

X = df[FEATURES].values

# ── Part 1A: Binary Party classification ─────────────────────────────────────
print("\n" + "="*60)
print("PART 1A — Binary Party Classification")
print("="*60)

y_party = df["party_enc"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y_party, test_size=0.2, random_state=42, stratify=y_party)

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.1),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

party_results, party_importances = {}, {}
for name, m in clf_models.items():
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    f1 = f1_score(y_te, pred, average="weighted")
    party_results[name] = f1
    party_importances[name] = np.abs(m.coef_[0]) if hasattr(m, "coef_") else m.feature_importances_
    print(f"\n── {name} ──  F1={f1:.4f}")
    print(classification_report(y_te, pred, target_names=le_party.classes_))

# ── Part 1B: Partisan Lean regression ────────────────────────────────────────
print("\n" + "="*60)
print("PART 1C — Partisan Lean Regression")
print("="*60)

y_lean = df["Partisan Lean"].values
X_trr, X_ter, y_trr, y_ter = train_test_split(X, y_lean, test_size=0.2, random_state=42)

reg_models = {
    "Linear Regression":   LinearRegression(),
    "Random Forest":       RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=200, random_state=42),
}

reg_results, reg_importances = {}, {}
for name, m in reg_models.items():
    m.fit(X_trr, y_trr)
    pred = m.predict(X_ter)
    mae = mean_absolute_error(y_ter, pred)
    r2  = r2_score(y_ter, pred)
    reg_results[name] = {"mae": mae, "r2": r2}
    reg_importances[name] = np.abs(m.coef_) if hasattr(m, "coef_") else m.feature_importances_
    print(f"\n── {name} ──  MAE={mae:.3f}  R²={r2:.4f}")

# ── Feature importance plots ──────────────────────────────────────────────────
def make_importance_fig(importances, title):
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(importances.keys()))
    for col, (name, imp) in enumerate(importances.items(), 1):
        order = np.argsort(imp)
        fig.add_trace(go.Bar(x=imp[order], y=[FEATURES[i] for i in order],
                             orientation="h", name=name), row=1, col=col)
    fig.update_layout(title=title, height=500, showlegend=False)
    return fig

fig_imp_party = make_importance_fig(party_importances, "Feature Importance — Party (Binary)")
fig_imp_reg   = make_importance_fig(reg_importances,   "Feature Importance — Partisan Lean (Regression)")

fig_imp_party.write_html(os.path.join(OUT_DIR, "party_feature_importance.html"))
fig_imp_reg.write_html(os.path.join(OUT_DIR, "lean_reg_feature_importance.html"))
print("\nFeature importance plots saved.")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STATIC MODEL SUMMARY")
print("="*60)
print(f"\n{'Model':<25} {'Party F1':>10}")
print("-"*37)
for name in clf_models:
    print(f"{name:<25} {party_results[name]:>10.4f}")
print(f"\n{'Model':<25} {'Lean MAE':>10} {'Lean R²':>10}")
print("-"*47)
for name, r in reg_results.items():
    print(f"{name:<25} {r['mae']:>10.3f} {r['r2']:>10.4f}")

# ── Part 2: Temporal sliding window — all targets ─────────────────────────────
print("\n" + "="*60)
print("PART 2 — Temporal Sliding Window (LR probe)")
print("="*60)

all_states   = np.array(sorted(df[STATE_COL].unique()))
n_states     = len(all_states)
rng          = np.random.RandomState(42)
perm         = rng.permutation(n_states)
split        = int(0.7 * n_states)
train_states = set(all_states[perm[:split]])
test_states  = set(all_states[perm[split:]])

probe_party = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
probe_reg   = LinearRegression()

months = pd.period_range(df["date"].min(), df["date"].max(), freq="M")
records = []

for month in months:
    mdf = df[df["date"].dt.to_period("M") == month].dropna(subset=FEATURES)
    if len(mdf) == 0:
        continue
    tr = mdf[mdf[STATE_COL].isin(train_states)]
    te = mdf[mdf[STATE_COL].isin(test_states)]
    if len(te) == 0:
        continue

    rec = {"month": str(month)}

    # Binary Party
    if len(tr["party_enc"].unique()) >= 2:
        probe_party.fit(tr[FEATURES], tr["party_enc"])
        pred = probe_party.predict(te[FEATURES])
        rec["party_f1"] = f1_score(te["party_enc"], pred, average="weighted", zero_division=0)

    # Regression Lean
    probe_reg.fit(tr[FEATURES], tr["Partisan Lean"])
    pred_r = probe_reg.predict(te[FEATURES])
    rec["lean_r2"] = r2_score(te["Partisan Lean"], pred_r)

    records.append(rec)

temp_df = pd.DataFrame(records)

# F1 plot (classification targets)
fig_f1 = go.Figure()
if "party_f1" in temp_df:
    fig_f1.add_trace(go.Scatter(x=temp_df["month"], y=temp_df["party_f1"],
                                mode="lines+markers", name="Party (binary)", line=dict(color="royalblue")))
fig_f1.update_layout(title="Party Predictability Over Time — F1 (LR probe)",
                     xaxis_title="Month", yaxis_title="Weighted F1",
                     yaxis=dict(range=[0, 1]), hovermode="x unified")
fig_f1.write_html(os.path.join(OUT_DIR, "party_temporal_f1.html"))

# R² plot (regression)
fig_r2 = go.Figure()
fig_r2.add_trace(go.Scatter(x=temp_df["month"], y=temp_df["lean_r2"],
                             mode="lines+markers", name="Partisan Lean (regression)", line=dict(color="seagreen")))
fig_r2.update_layout(title="Partisan Lean Regression Over Time — R² (LR probe)",
                     xaxis_title="Month", yaxis_title="R²",
                     hovermode="x unified")
fig_r2.write_html(os.path.join(OUT_DIR, "party_temporal_r2.html"))

print(f"\nTemporal F1 plot saved: outputs/party_temporal_f1.html")
print(f"Temporal R² plot saved: outputs/party_temporal_r2.html")

print("\nTop-5 months — Party F1:")
print(temp_df.nlargest(5, "party_f1")[["month", "party_f1"]].to_string(index=False))
print("\nTop-5 months — Lean R²:")
print(temp_df.nlargest(5, "lean_r2")[["month", "lean_r2"]].to_string(index=False))

fig_f1.show()
fig_r2.show()
