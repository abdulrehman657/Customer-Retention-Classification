"""
Customer Health Checker - Streamlit front end wired to the trained
random_forest_health_model.pkl (see Model-Training.py).

Read-only viewer: pick a random real customer and see the AI's verdict on
them - Healthy, At Risk, or Churned - with a confidence score and the real
data behind it. No editable controls; this is purely for looking at what
the model sees, not for simulating hypothetical customers.
"""

import math
import random
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# =========================================================
# 1. Config, constants, encodings
# =========================================================

st.set_page_config(
    page_title="Customer Health Checker · Random Forest AI",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Resolved relative to this file (not the working directory) so the app
# finds its files no matter where `streamlit run` is invoked from - this
# folder holds App.py, the CSV, and requirements.txt; the model lives one
# level up in the sibling "Model-Training" folder. Folder names are
# hyphenated, not spaced - a space in "Front End" broke Streamlit Community
# Cloud's requirements.txt path resolution (it silently truncated the path
# at the space), so spaces are avoided entirely here.
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR.parent / "Model-Training" / "random_forest_health_model.pkl"
DATA_PATH = APP_DIR / "balanced_customer_health_dataset.csv"
POPULATION_SIZE = 140
POPULATION_SEED = 42

# Exact column order the model was trained on (Model-Training.py) - order matters.
FEATURE_ORDER = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
    "Monthly_Spend_Rate",
    "Interaction_To_Usage_Ratio",
    "Lifecycle_Ratio",
    "Spend_Per_Interaction",
]

# Plain-language translation of each engineered feature, shown in the
# "What The AI Looks At Most" panel so non-technical readers understand
# what each signal actually represents. "Age" is deliberately left out of
# that panel (not this dict) at the user's request.
FEATURE_DESCRIPTIONS = {
    "Age": "The customer's age",
    "Gender": "The customer's gender",
    "Tenure": "How long they've been a customer",
    "Usage Frequency": "How often they use the product",
    "Subscription Type": "Which plan tier they're on",
    "Contract Length": "How long a contract they've committed to",
    "Total Spend": "Total amount they've spent with us so far",
    "Last Interaction": "Days since they last engaged with us",
    "Monthly_Spend_Rate": "Revenue per month, relative to how long they've been a customer",
    "Interaction_To_Usage_Ratio": "How often they're in touch with us relative to how often they use the product",
    "Lifecycle_Ratio": "How mature the relationship is, relative to the customer's age",
    "Spend_Per_Interaction": "How much value they generate each time they engage with us",
}

# Confirmed against the trained model: model.classes_ == [0, 1, 2], and class
# 0 has the lowest avg Payment Delay / Support Calls in the training data,
# class 2 the highest -> 0=Healthy, 1=At Risk, 2=Churned.
CLASS_META = {
    0: {
        "label": "Healthy",
        "color": "var(--emerald)",
        "bg": "oklch(0.75 0.16 160 / 10%)",
        "ring": "oklch(0.75 0.16 160 / 45%)",
        "icon": "shield",
        "dot": "🟢",
        "explanation": "This customer looks stable and isn't showing signs of leaving.",
    },
    1: {
        "label": "At Risk",
        "color": "var(--amber)",
        "bg": "oklch(0.82 0.15 80 / 10%)",
        "ring": "oklch(0.82 0.15 80 / 45%)",
        "icon": "triangle",
        "dot": "🟡",
        "explanation": "This customer is showing early warning signs - worth reaching out.",
    },
    2: {
        "label": "Churned",
        "color": "var(--crimson)",
        "bg": "oklch(0.66 0.22 18 / 10%)",
        "ring": "oklch(0.66 0.22 18 / 45%)",
        "icon": "trendDown",
        "dot": "🔴",
        "explanation": "The AI considers this customer already lost, or about to be.",
    },
}

# Inline SVG icons - no external requests, renders reliably everywhere.
ICONS = {
    "cpu": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/></svg>',
    "shield": '<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>',
    "triangle": '<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4M12 17h.01"/></svg>',
    "trendDown": '<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 17h6v-6"/><path d="m22 17-8.5-8.5-5 5L2 7"/></svg>',
    "calendar": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/></svg>',
    "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    "wallet": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 7V4a1 1 0 0 0-1-1H5a2 2 0 0 0 0 4h15a1 1 0 0 1 1 1v4h-3a2 2 0 0 0 0 4h3a1 1 0 0 0 1-1v-2a1 1 0 0 0-1-1"/><path d="M3 5v14a2 2 0 0 0 2 2h15a1 1 0 0 0 1-1v-4"/></svg>',
    "clock": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>',
    "alertTriangle": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4M12 17h.01"/></svg>',
    "user": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/></svg>',
}

# The raw categorical columns in the CSV are already label-encoded (0/1/2)
# from an upstream preprocessing step that isn't part of this repo. These
# maps are used both to show human-readable dropdowns and to decode a
# loaded real customer's values for display - assumed alphabetical (sklearn
# LabelEncoder default): Female/Male, Basic/Premium/Standard, Annual/Monthly/Quarterly.
GENDER_MAP = {0: "Female", 1: "Male"}
SUBSCRIPTION_MAP = {0: "Basic", 1: "Premium", 2: "Standard"}
CONTRACT_MAP = {0: "Annual", 1: "Monthly", 2: "Quarterly"}

BADGE_STYLE = {
    "Trigger App Push Engagement": (
        "background: oklch(0.78 0.14 210 / 12%); color: var(--neon); "
        "box-shadow: inset 0 0 0 1px oklch(0.78 0.14 210 / 30%);"
    ),
    "Issue Value Credits": (
        "background: oklch(0.6 0.18 290 / 15%); color: oklch(0.78 0.14 290); "
        "box-shadow: inset 0 0 0 1px oklch(0.6 0.18 290 / 30%);"
    ),
    "Billing Grace Period Outreach": (
        "background: oklch(0.82 0.15 80 / 12%); color: var(--amber); "
        "box-shadow: inset 0 0 0 1px oklch(0.82 0.15 80 / 30%);"
    ),
}

# Editable fields shown as plain sliders, in real units (no multipliers).
# Bounds taken from the actual min/max in balanced_customer_health_dataset.csv.
FIELD_CONFIG = [
    {
        "key": "Tenure",
        "label": "Months As A Customer",
        "icon": "calendar",
        "accent": "var(--neon)",
        "min": 1,
        "max": 60,
        "step": 1,
        "suffix": " mo",
        "help": "How long they've been with us.",
    },
    {
        "key": "Usage Frequency",
        "label": "How Often They Use The Product",
        "icon": "activity",
        "accent": "var(--emerald)",
        "min": 1,
        "max": 30,
        "step": 1,
        "suffix": "x / month",
        "help": "Times per month they actively use it.",
    },
    {
        "key": "Total Spend",
        "label": "Total Amount Spent",
        "icon": "wallet",
        "accent": "var(--amber)",
        "min": 100,
        "max": 1000,
        "step": 10,
        "suffix": "",
        "prefix": "$",
        "help": "Lifetime amount they've spent with us so far.",
    },
    {
        "key": "Last Interaction",
        "label": "Days Since Last Contact",
        "icon": "clock",
        "accent": "var(--neon)",
        "min": 0,
        "max": 30,
        "step": 1,
        "suffix": " days ago",
        "help": "How long since they last engaged with us.",
    },
    {
        "key": "Payment Delay",
        "label": "Days Late On Payment",
        "icon": "alertTriangle",
        "accent": "var(--crimson)",
        "min": 0,
        "max": 30,
        "step": 1,
        "suffix": " days late",
        "help": "The AI's health verdict doesn't use this (it wasn't part of the "
        "training data) - it only changes the recommended action below.",
    },
]


# =========================================================
# 2. Cached data / model loading
# =========================================================


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_population(n=POPULATION_SIZE, seed=POPULATION_SEED):
    df = pd.read_csv(DATA_PATH)
    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    sample.insert(0, "CustomerID", [f"CX-{10234 + i}" for i in range(len(sample))])
    return sample


model = load_model()
population = load_population()


@st.cache_data
def get_feature_importance():
    # "Age" is deliberately excluded from this panel - not shown to the user
    # as a signal, even though the underlying model still uses it internally.
    importances = pd.Series(model.feature_importances_, index=FEATURE_ORDER).drop("Age")
    top = importances.sort_values(ascending=False).head(4)
    max_val = top.max() or 1
    return [
        {
            "label": name.replace("_", " "),
            "description": FEATURE_DESCRIPTIONS.get(name, ""),
            "pct": val * 100,
            "bar_pct": val / max_val * 100,
        }
        for name, val in top.items()
    ]


@st.cache_data
def population_baseline_counts():
    """Real predicted class counts across the 140 sample customers' own
    original values - a small, static context stat, not recomputed live."""
    pred = model.predict(population[FEATURE_ORDER])
    counts = pd.Series(pred).value_counts().reindex([0, 1, 2], fill_value=0)
    return counts


# =========================================================
# 3. Feature engineering + real inference for a single customer
# =========================================================


def engineer_one(values: dict) -> dict:
    """Recompute the 4 engineered ratio features using the exact same
    formulas as the dataset's precomputed columns (verified against
    balanced_customer_health_dataset.csv)."""
    total_spend = values["Total Spend"]
    tenure = values["Tenure"]
    usage = values["Usage Frequency"]
    last_interaction = values["Last Interaction"]
    age = values["Age"]
    return {
        "Monthly_Spend_Rate": total_spend / (tenure + 1),
        "Interaction_To_Usage_Ratio": usage / (last_interaction + 1),
        "Lifecycle_Ratio": tenure / (age + 1),
        "Spend_Per_Interaction": total_spend / (usage + 1),
    }


def strategy_for(values: dict, engineered: dict) -> str:
    if values["Payment Delay"] >= 12:
        return "Billing Grace Period Outreach"
    if values["Last Interaction"] >= 18:
        return "Trigger App Push Engagement"
    if engineered["Spend_Per_Interaction"] >= 60:
        return "Issue Value Credits"
    return "Trigger App Push Engagement"


def predict_customer(values: dict) -> dict:
    engineered = engineer_one(values)
    row = {**values, **engineered}
    X = pd.DataFrame([row])[FEATURE_ORDER]
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    class_index = {c: i for i, c in enumerate(model.classes_)}
    confidence = float(proba[class_index[pred]])
    return {
        "pred": pred,
        "confidence": confidence,
        "engineered": engineered,
        "strategy": strategy_for(values, engineered),
    }


# =========================================================
# 4. Session state - which customer is currently shown
# =========================================================

FIELD_KEYS = [c["key"] for c in FIELD_CONFIG] + ["Age", "Gender", "Subscription Type", "Contract Length"]

if "active_customer_id" not in st.session_state:
    st.session_state.active_customer_id = population.iloc[0]["CustomerID"]


def pick_random_customer():
    st.session_state.active_customer_id = random.choice(population["CustomerID"].tolist())


active_row = population.loc[population["CustomerID"] == st.session_state.active_customer_id].iloc[0]
current_values = {key: (active_row[key].item() if hasattr(active_row[key], "item") else active_row[key]) for key in FIELD_KEYS}
result = predict_customer(current_values)
meta = CLASS_META[result["pred"]]


# =========================================================
# 5. Styling
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap');

:root {
  --background: #090d16;
  --foreground: oklch(0.96 0.01 240);
  --muted-foreground: oklch(0.65 0.02 240);
  --emerald: oklch(0.75 0.16 160);
  --amber: oklch(0.82 0.15 80);
  --crimson: oklch(0.66 0.22 18);
  --neon: oklch(0.78 0.14 210);
  --glass-border: oklch(0.75 0.06 235 / 14%);
}

html, body, [class*="css"] { font-family: 'Geist', ui-sans-serif, system-ui, sans-serif; }
.stApp {
  background:
    radial-gradient(circle at 15% -10%, oklch(0.3 0.06 235 / 35%), transparent 45%),
    radial-gradient(circle at 100% 0%, oklch(0.25 0.08 290 / 20%), transparent 40%),
    var(--background);
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 720px; }

.font-mono { font-family: 'Geist Mono', ui-monospace, monospace; }
.muted { color: var(--muted-foreground); }

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
.glass-card, .verdict-card, .header-wrap { animation: fadeUp 0.35s ease-out; }

.glass-card {
  background: linear-gradient(160deg, oklch(0.26 0.03 250 / 55%), oklch(0.16 0.03 250 / 45%));
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid var(--glass-border);
  border-radius: 1.1rem;
  padding: 1.4rem 1.6rem;
  box-shadow: 0 12px 32px -18px oklch(0 0 0 / 60%);
}
.card-title {
  display:flex; align-items:center; gap:0.55rem;
  font-weight: 600; font-size: 0.98rem; margin-bottom: 1.15rem;
}
.card-title .icon-chip {
  display:grid; place-items:center; width:1.9rem; height:1.9rem; border-radius:0.55rem;
  background: oklch(1 0 0 / 6%); flex-shrink:0;
}

/* ---------- Header ---------- */
.header-wrap { display:flex; flex-direction:column; align-items:center; gap:0.6rem; text-align:center; }
.header-badge {
  display:grid; place-items:center; width:3.4rem; height:3.4rem; border-radius:1rem;
  color: var(--neon); background: oklch(0.78 0.14 210 / 12%);
  box-shadow: inset 0 0 0 1px oklch(0.78 0.14 210 / 30%), 0 0 30px -8px oklch(0.78 0.14 210 / 45%);
}
.header-title { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; }
.intro-text { font-size: 0.92rem; line-height: 1.6; color: var(--muted-foreground); text-align: center; max-width: 34rem; }

/* ---------- Picker row ---------- */
.picker-caption {
  display:flex; align-items:center; justify-content:center; gap:0.4rem;
  text-align:center; font-size:0.8rem; color: var(--muted-foreground); margin-top:0.7rem;
}
.picker-caption .id-chip {
  font-family:'Geist Mono',monospace; font-size:0.76rem; color: var(--foreground);
  background: oklch(1 0 0 / 6%); border: 1px solid oklch(1 0 0 / 10%);
  border-radius: 0.4rem; padding: 0.1rem 0.5rem;
}

/* ---------- Verdict card ---------- */
.verdict-card {
  border-radius: 1.4rem;
  padding: 2.2rem 1.75rem;
  text-align: center;
  border: 1px solid var(--glass-border);
  background: linear-gradient(160deg, oklch(0.24 0.03 250 / 60%), oklch(0.15 0.03 250 / 50%));
  transition: box-shadow 0.3s ease;
}
.confidence-ring-wrap { position: relative; display:inline-grid; place-items:center; margin-bottom: 0.4rem; }
.confidence-ring-wrap svg { display:block; }
.confidence-ring-center {
  position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap: 0.15rem;
}
.confidence-pct { font-family:'Geist Mono',monospace; font-size:1.5rem; font-weight:700; letter-spacing:-0.01em; }
.confidence-caption { font-size: 0.62rem; text-transform:uppercase; letter-spacing:0.1em; color: var(--muted-foreground); }

.verdict-label { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.01em; margin-top: 0.9rem; }
.verdict-explanation { font-size: 0.95rem; color: var(--muted-foreground); margin-top: 0.55rem; max-width: 30rem; margin-left: auto; margin-right: auto; line-height: 1.5; }

.action-row { display:flex; align-items:center; justify-content:center; gap:0.5rem; margin-top:1.3rem; flex-wrap:wrap; }
.badge-pill { display:inline-flex; align-items:center; gap:6px; border-radius:9999px; padding:0.4rem 0.9rem; font-size:0.78rem; font-weight:600; }

/* ---------- Editable fields ---------- */
.field-row { display:flex; align-items:center; justify-content:space-between; margin-top: 1.1rem; gap: 0.75rem; }
.field-row:first-of-type { margin-top: 0; }
.field-label-group { display:flex; align-items:center; gap:0.6rem; }
.field-icon-chip {
  display:grid; place-items:center; width:1.75rem; height:1.75rem; border-radius:0.5rem;
  background: oklch(1 0 0 / 5%); flex-shrink:0;
}
.field-label { font-size: 0.87rem; font-weight: 500; }
.field-value-badge {
  font-family:'Geist Mono',monospace; font-size:0.78rem; font-weight:600;
  border:1px solid oklch(1 0 0/10%); background:oklch(1 0 0/5%);
  border-radius:0.4rem; padding:0.15rem 0.55rem; white-space:nowrap;
}
.field-caption { font-size: 0.76rem; color: var(--muted-foreground); margin: 0.15rem 0 0.9rem 2.35rem; line-height:1.4; }
.field-caption.warn { color: var(--amber); }

/* ---------- Feature importance ---------- */
.fi-row { margin-bottom:0.9rem; }
.fi-top { display:flex; justify-content:space-between; margin-bottom:0.2rem; font-family:'Geist Mono',monospace; font-size:0.78rem; }
.fi-desc { font-size: 0.72rem; color: var(--muted-foreground); margin: -0.05rem 0 0.35rem 0; }
.fi-track { height:0.5rem; width:100%; border-radius:9999px; background:oklch(1 0 0/6%); overflow:hidden; }
.fi-fill { height:100%; border-radius:9999px; background:var(--neon); box-shadow:0 0 10px -1px var(--neon); }

/* ---------- Context strip ---------- */
.context-strip {
  display:flex; align-items:center; justify-content:center; gap:0.6rem;
  flex-wrap:wrap; margin-top: 1.75rem;
}
.stat-pill {
  display:inline-flex; align-items:center; gap:0.4rem;
  font-size:0.8rem; color: var(--muted-foreground);
  background: oklch(1 0 0 / 4%); border: 1px solid oklch(1 0 0 / 8%);
  border-radius: 9999px; padding: 0.35rem 0.85rem;
}
.stat-pill b { color: var(--foreground); font-family:'Geist Mono',monospace; }

/* ---------- Buttons ---------- */
.stButton>button {
  border-radius: 0.8rem; font-weight: 600; transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover { transform: translateY(-1px); }
div[data-testid="stExpander"] {
  border-radius: 1rem !important; border-color: var(--glass-border) !important;
  background: oklch(1 0 0 / 2%);
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 6. Header + intro
# =========================================================

st.markdown(
    f"""
    <div class="header-wrap">
      <div class="header-badge">{ICONS['cpu']}</div>
      <div class="header-title">Customer Health Checker</div>
      <p class="intro-text">This runs a real trained AI model (a Random Forest) - not a mockup.
      Browse real customers from the dataset below and see exactly how the AI reads
      each one - Healthy, At Risk, or Churned.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# =========================================================
# 7. Customer picker
# =========================================================

st.button("🎲  Show Me Another Real Customer", on_click=pick_random_customer, use_container_width=True, type="primary")

st.markdown(
    f'<div class="picker-caption">Showing real customer '
    f'<span class="id-chip">{st.session_state.active_customer_id}</span> from the dataset.</div>',
    unsafe_allow_html=True,
)

st.write("")

# =========================================================
# 8. Verdict card - real confidence ring (SVG), colored to match status
# =========================================================

size, stroke = 148, 12
radius = (size - stroke) / 2
circumference = 2 * math.pi * radius
pct = result["confidence"] * 100
offset = circumference - (pct / 100) * circumference

st.markdown(
    f"""
    <div class="verdict-card" style="box-shadow: 0 0 0 1px {meta['ring']}, 0 0 46px -12px {meta['ring']};">
      <div class="confidence-ring-wrap" style="width:{size}px;height:{size}px;">
        <svg width="{size}" height="{size}" style="transform:rotate(-90deg)">
          <circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none" stroke="oklch(0.4 0.03 250 / 35%)" stroke-width="{stroke}"></circle>
          <circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none" stroke="{meta['color']}" stroke-width="{stroke}" stroke-linecap="round"
            stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
            style="transition: stroke-dashoffset 0.6s cubic-bezier(0.22,1,0.36,1); filter: drop-shadow(0 0 8px {meta['ring']});"></circle>
        </svg>
        <div class="confidence-ring-center" style="color:{meta['color']}">
          {ICONS[meta['icon']]}
        </div>
      </div>
      <div class="verdict-label" style="color:{meta['color']}">{meta['label']}</div>
      <div class="verdict-explanation">{meta['explanation']}</div>
      <div class="confidence-caption" style="margin-top:0.9rem;">AI Confidence · {pct:.0f}%</div>
    </div>
    """,
    unsafe_allow_html=True,
)

strategy = result["strategy"]
st.markdown(
    f'<div class="action-row"><span class="muted" style="font-size:0.82rem;">Recommended action:</span>'
    f'<span class="badge-pill" style="{BADGE_STYLE[strategy]}">{strategy}</span></div>',
    unsafe_allow_html=True,
)

st.write("")
st.write("")

# =========================================================
# 9. Customer info - read-only, no editable controls
# =========================================================

st.markdown(
    f'<div class="card-title"><span class="icon-chip" style="color:var(--neon)">{ICONS["user"]}</span>Who They Are</div>',
    unsafe_allow_html=True,
)
who_rows = [
    ("Age", str(current_values["Age"])),
    ("Gender", GENDER_MAP[current_values["Gender"]]),
    ("Plan", SUBSCRIPTION_MAP[current_values["Subscription Type"]]),
    ("Contract Length", CONTRACT_MAP[current_values["Contract Length"]]),
]
who_html = "".join(
    f'<div class="field-row" style="margin-top:{"0" if i == 0 else "0.9rem"};">'
    f'<span class="field-label">{label}</span>'
    f'<span class="field-value-badge">{value}</span></div>'
    for i, (label, value) in enumerate(who_rows)
)
st.markdown(f'<div class="glass-card">{who_html}</div>', unsafe_allow_html=True)

st.write("")

st.markdown(
    f'<div class="card-title"><span class="icon-chip" style="color:var(--emerald)">{ICONS["activity"]}</span>How They Behave</div>',
    unsafe_allow_html=True,
)
behave_html = ""
for i, cfg in enumerate(FIELD_CONFIG):
    value = current_values[cfg["key"]]
    shown = f"{cfg.get('prefix', '')}{value:,.0f}{cfg['suffix']}"
    warn_class = " warn" if cfg["key"] == "Payment Delay" else ""
    behave_html += (
        f'<div class="field-row" style="margin-top:{"0" if i == 0 else "1.1rem"};">'
        f'<div class="field-label-group">'
        f'<span class="field-icon-chip" style="color:{cfg["accent"]}">{ICONS[cfg["icon"]]}</span>'
        f'<span class="field-label">{cfg["label"]}</span></div>'
        f'<span class="field-value-badge" style="color:{cfg["accent"]}">{shown}</span>'
        "</div>"
        f'<div class="field-caption{warn_class}">{cfg["help"]}</div>'
    )
st.markdown(f'<div class="glass-card">{behave_html}</div>', unsafe_allow_html=True)

st.write("")

# =========================================================
# 10. Why the AI decided this (global signal importance, simplified)
# =========================================================

with st.expander("🧠  What does the AI generally look at most?"):
    st.markdown(
        '<p class="muted" style="font-size:0.8rem; margin-bottom:1rem;">'
        "These are the signals the AI relies on most heavily across all customers in "
        "general - not a customer-specific breakdown of this one verdict.</p>",
        unsafe_allow_html=True,
    )
    fi_html = ""
    for f in get_feature_importance():
        fi_html += (
            f'<div class="fi-row"><div class="fi-top"><span>{f["label"]}</span>'
            f'<span class="muted">{f["pct"]:.1f}%</span></div>'
            f'<div class="fi-desc">{f["description"]}</div>'
            f'<div class="fi-track"><div class="fi-fill" style="width:{f["bar_pct"]}%"></div></div></div>'
        )
    st.markdown(fi_html, unsafe_allow_html=True)

# =========================================================
# 11. Small context strip - population-level stat, static
# =========================================================

counts = population_baseline_counts()
stat_pills = "".join(
    f'<span class="stat-pill">{CLASS_META[k]["dot"]} <b>{int(counts[k])}</b> {CLASS_META[k]["label"]}</span>'
    for k in (0, 1, 2)
)
st.markdown(
    f'<div class="context-strip"><span class="stat-pill">Across <b>{int(counts.sum())}</b> real sample customers</span>{stat_pills}</div>',
    unsafe_allow_html=True,
)
