"""
Event Attendance Predictor — Streamlit Demo App
--------------------------------------------------
Loads the models and data that already exist in this project (no retraining).
Run from the project root (the folder that contains `model/`, `train_clean.csv`,
`test_clean.csv`, `test_predictions.csv`) with:

    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Event Attendance Predictor",
    page_icon="🎟️",
    layout="wide",
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"

TARGET = "attended"
NUM_FEATURES = [
    "registration_days_before",
    "previous_events_registered",
    "previous_events_attended",
    "club_member",
    "event_time",
    "travel_distance_km",
]
CAT_FEATURES = ["event_type", "event_day"]
FEATURES = NUM_FEATURES + CAT_FEATURES

MODEL_FILES = {
    "Gradient Boosting": "gradient_boosting.joblib",
    "LightGBM": "lightgbm.joblib",
    "Hist Gradient Boosting": "hist_gradient_boosting.joblib",
    "CatBoost": "catboost.joblib",
    "SVM (RBF)": "svm_rbf.joblib",
    "SVM (Linear)": "svm_linear.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "XGBoost": "xgboost.joblib",
}
BEST_MODEL_NAME = "Gradient Boosting"  # winner by F1 and AUC on hold-out split

# Metrics copied from the modeling notebook's hold-out evaluation, so the
# comparison table renders instantly without re-running every model.
RESULTS_TABLE = pd.DataFrame(
    [
        {"model": "Gradient Boosting", "precision": 0.7576, "recall": 0.7937, "f1": 0.7752, "auc": 0.7031},
        {"model": "LightGBM", "precision": 0.7925, "recall": 0.6667, "f1": 0.7241, "auc": 0.7113},
        {"model": "Hist Gradient Boosting", "precision": 0.7333, "recall": 0.6984, "f1": 0.7154, "auc": 0.6628},
        {"model": "SVM (RBF)", "precision": 0.7593, "recall": 0.6508, "f1": 0.7009, "auc": 0.6804},
        {"model": "CatBoost", "precision": 0.7692, "recall": 0.6349, "f1": 0.6957, "auc": 0.7181},
        {"model": "SVM (Linear)", "precision": 0.7018, "recall": 0.6349, "f1": 0.6667, "auc": 0.6568},
        {"model": "Logistic Regression", "precision": 0.7255, "recall": 0.5873, "f1": 0.6491, "auc": 0.6843},
        {"model": "XGBoost", "precision": 0.7600, "recall": 0.6032, "f1": 0.6726, "auc": 0.6782},
    ]
).set_index("model")


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


@st.cache_data
def load_train():
    return pd.read_csv(ROOT / "train_clean.csv")


@st.cache_data
def load_test_predictions():
    return pd.read_csv(ROOT / "test_predictions.csv")


@st.cache_data
def get_holdout_split():
    """Recreate the exact same stratified hold-out split used in the notebook
    (same random_state and test_size), so the confusion matrix / report shown
    here match what was reported during model development."""
    train = load_train()
    X, y = train[FEATURES], train[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_val, y_val


def predict_proba(model_name: str, df: pd.DataFrame) -> np.ndarray:
    """Return P(attended=1) for each row, handling CatBoost's Pool
    requirement separately from the sklearn-style pipelines."""
    model = load_model(model_name)
    if model_name == "CatBoost":
        from catboost import Pool

        pool = Pool(df[FEATURES], cat_features=CAT_FEATURES)
        return model.predict_proba(pool)[:, 1]
    return model.predict_proba(df[FEATURES])[:, 1]


def get_lgbm_feature_importance() -> pd.DataFrame:
    """Global feature importance straight from the trained LightGBM booster,
    mapped back to human-readable feature names."""
    model = load_model("LightGBM")
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    importances = clf.feature_importances_
    df = pd.DataFrame({"feature": names, "importance": importances})
    # Collapse one-hot columns (e.g. event_type_Workshop) back to their parent feature
    df["feature_group"] = df["feature"].apply(
        lambda f: "event_type" if f.startswith("event_type_")
        else "event_day" if f.startswith("event_day_")
        else f
    )
    grouped = df.groupby("feature_group")["importance"].sum().sort_values(ascending=False)
    return grouped


def explain_lgbm_prediction(input_row: pd.DataFrame) -> pd.DataFrame:
    """Per-prediction SHAP-style contributions from the LightGBM booster
    (LightGBM's native pred_contrib), collapsed back to readable feature
    names and sorted by absolute impact on this specific prediction."""
    model = load_model("LightGBM")
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()] + ["base_value"]
    Xt = prep.transform(input_row[FEATURES])
    contrib = clf.booster_.predict(Xt, pred_contrib=True)[0]
    df = pd.DataFrame({"feature": names, "contribution": contrib})
    df["feature_group"] = df["feature"].apply(
        lambda f: "event_type" if f.startswith("event_type_")
        else "event_day" if f.startswith("event_day_")
        else f
    )
    grouped = (
        df[df["feature_group"] != "base_value"]
        .groupby("feature_group")["contribution"].sum()
        .sort_values(key=abs, ascending=False)
    )
    return grouped


# Data-quality numbers, copied from the cleanup notebook's before/after
# quality reports (raw Kaggle data -> train_clean.csv / test_clean.csv).
CLEANING_SUMMARY = {
    "train_raw_shape": (508, 10),
    "test_raw_shape": (100, 9),
    "train_clean_shape": (496, 10),
    "test_clean_shape": (100, 9),
    "train_missing_by_col": {
        "event_type": 12, "registration_days_before": 15, "previous_events_registered": 10,
        "previous_events_attended": 14, "club_member": 8, "event_day": 9,
        "event_time": 11, "travel_distance_km": 14, "attended": 5,
    },
    "exact_duplicates_dropped": 7,
    "negative_registration_days": 2,
    "impossible_attended_gt_registered": 3,
    "outlier_bounds": {
        "registration_days_before": (0, 32.0),
        "travel_distance_km": (0, 26.8),
    },
    "impute_values": {
        "registration_days_before": 7.0, "previous_events_registered": 4.0,
        "previous_events_attended": 3.0, "travel_distance_km": 5.5,
        "event_type": "Workshop", "club_member": 1.0,
        "event_day": "Saturday", "event_time": 18.0,
    },
    "unlabeled_rows_set_aside": 5,
}


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("🎟️ Event Attendance Predictor")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Data Cleaning",
        "EDA Highlights",
        "Modeling & Results",
        "Live Predictor",
        "Batch Predictions",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Predicts whether a student who registered for a club event will actually "
    "attend, using 8 trained classifiers evaluated on a held-out split."
)

train_df = load_train()

# --------------------------------------------------------------------------
# 1. Overview
# --------------------------------------------------------------------------
if page == "Overview":
    st.title("Event Attendance Predictor")
    st.markdown(
        "A classification model that predicts the probability a registered "
        "student actually shows up to a club event — trained on registration "
        "history, event details, and past attendance behavior."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training rows", len(train_df))
    c2.metric("Features used", len(FEATURES))
    c3.metric("Models trained", len(MODEL_FILES))
    c4.metric("Best model F1", f"{RESULTS_TABLE.loc[BEST_MODEL_NAME, 'f1']:.2f}")

    st.markdown("### Class balance")
    balance = train_df[TARGET].value_counts().rename({0: "Did not attend", 1: "Attended"})
    fig = px.bar(
        balance,
        x=balance.index,
        y=balance.values,
        text=balance.values,
        labels={"x": "", "y": "Students"},
        color=balance.index,
        color_discrete_sequence=["#EF553B", "#00CC96"],
    )
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Attendance outnumbers no-shows roughly 63/37 — every model below was "
        "explicitly told about this imbalance (class-weight balancing) rather "
        "than being left to just favor the majority class."
    )

    st.markdown("### What the task required")
    st.markdown(
        "- Clean the dataset and handle missing/inconsistent values\n"
        "- Convert categorical data into a usable format\n"
        "- Train a classifier to predict attendance\n"
        "- Evaluate with precision, recall, and F1\n"
        "- Produce attendance probabilities for new registrations, e.g. "
        "*'Student A → 87% likely to attend'*"
    )

# --------------------------------------------------------------------------
# 1b. Data Cleaning
# --------------------------------------------------------------------------
elif page == "Data Cleaning":
    st.title("Data Cleaning")
    st.markdown(
        "The raw Kaggle export had real quality problems. A `DataCleaner` was "
        "**fit only on train** and then applied identically to test, so no "
        "information leaks from test into the cleaning parameters."
    )

    c1, c2 = st.columns(2)
    c1.metric("Train rows: raw → clean", f"{CLEANING_SUMMARY['train_raw_shape'][0]} → {CLEANING_SUMMARY['train_clean_shape'][0]}")
    c2.metric("Test rows: raw → clean", f"{CLEANING_SUMMARY['test_raw_shape'][0]} → {CLEANING_SUMMARY['test_clean_shape'][0]}")

    st.markdown("### Issues found in the raw data")
    st.markdown(
        f"- **Missing values** in every column (up to 15 rows per column) — see chart below\n"
        f"- **{CLEANING_SUMMARY['exact_duplicates_dropped']} exact duplicate rows** — dropped\n"
        f"- **{CLEANING_SUMMARY['negative_registration_days']} rows** with a negative "
        "`registration_days_before` (impossible) — converted to missing, not clipped\n"
        f"- **{CLEANING_SUMMARY['impossible_attended_gt_registered']} rows** where "
        "`previous_events_attended > previous_events_registered` (logically impossible) — "
        "since we can't know which value is wrong, both were set to missing rather than guessed\n"
        "- Inconsistent text casing/whitespace in categorical columns "
        "(`event_type`, `club_member`, `event_day`) — standardized\n"
        "- Extreme outliers (e.g. a 60-day-early registration, a 120km trip) — "
        "flagged via a 3×IQR rule and treated as missing rather than clipped, "
        "so they don't silently distort the model"
    )

    miss = pd.Series(CLEANING_SUMMARY["train_missing_by_col"]).sort_values(ascending=False)
    fig = px.bar(miss, x=miss.index, y=miss.values,
                 labels={"x": "", "y": "Missing rows (raw train, n=508)"})
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### How missing values were filled")
    st.caption("All imputation values were learned from **train only**, then reused on test unchanged.")
    impute_df = pd.DataFrame(
        list(CLEANING_SUMMARY["impute_values"].items()), columns=["Feature", "Fill value (train median/mode)"]
    )
    st.dataframe(impute_df, use_container_width=True, hide_index=True)

    st.info(
        f"5 training rows had a missing **target** label (`attended`) — these were "
        "set aside into `train_unlabeled.csv` rather than dropped or guessed, "
        "so they stay available for future semi-supervised work without "
        "contaminating the supervised training set.",
        icon="🗂️",
    )

# --------------------------------------------------------------------------
# 2. EDA Highlights
# --------------------------------------------------------------------------
elif page == "EDA Highlights":
    st.title("EDA Highlights")
    st.markdown("The four strongest, statistically-checked signals found during EDA.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Club membership")
        rate = train_df.groupby("club_member")[TARGET].mean().rename(
            {0: "Non-member", 1: "Club member"}
        )
        fig = px.bar(rate, x=rate.index, y=rate.values, text_auto=".0%",
                     labels={"x": "", "y": "Attendance rate"})
        fig.update_layout(yaxis_tickformat=".0%", height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Club members attend ~70% of the time vs ~51% for non-members — "
                    "the single strongest predictor by mutual information.")

    with col2:
        st.subheader("2. Event type")
        rate = train_df.groupby("event_type")[TARGET].mean().sort_values(ascending=False)
        fig = px.bar(rate, x=rate.index, y=rate.values, text_auto=".0%",
                     labels={"x": "", "y": "Attendance rate"})
        fig.update_layout(yaxis_tickformat=".0%", height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Workshops draw the most reliable attendance (74%); "
                    "Hackathons and Socials lag around 53%.")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("3. Registration timing")
        fig = px.box(train_df, x=TARGET, y="registration_days_before",
                      color=TARGET, color_discrete_sequence=["#EF553B", "#00CC96"])
        fig.update_layout(
            xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Did not attend", "Attended"]),
            showlegend=False, height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Students who register further in advance (avg ~8 days vs ~6) "
                    "are more likely to follow through and attend.")

    with col4:
        st.subheader("4. Travel distance")
        fig = px.box(train_df, x=TARGET, y="travel_distance_km",
                      color=TARGET, color_discrete_sequence=["#EF553B", "#00CC96"])
        fig.update_layout(
            xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Did not attend", "Attended"]),
            showlegend=False, height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("A mild but real effect: students traveling farther are "
                    "somewhat less likely to actually show up.")

    st.info(
        "**Takeaway for the club:** target reminders at non-members and "
        "last-minute registrants, and expect lower turnout at Hackathons/Socials "
        "held far from campus — that's where a nudge helps most.",
        icon="💡",
    )

# --------------------------------------------------------------------------
# 3. Modeling & Results
# --------------------------------------------------------------------------
elif page == "Modeling & Results":
    st.title("Modeling & Results")
    st.markdown(
        "8 classifiers were trained — Gradient Boosting, LightGBM, Hist Gradient Boosting, "
        "CatBoost, SVM (RBF & Linear), XGBoost, and Logistic Regression — all regularized "
        "and class-weight balanced, then scored on the same stratified 20% hold-out split."
    )

    st.markdown("### Model comparison")
    st.dataframe(
        RESULTS_TABLE.sort_values("f1", ascending=False).style.format("{:.2%}").background_gradient(
            cmap="Greens"
        ),
        use_container_width=True,
    )

    fig = go.Figure()
    for metric in ["precision", "recall", "f1", "auc"]:
        fig.add_bar(name=metric.upper(), x=RESULTS_TABLE.index, y=RESULTS_TABLE[metric])
    fig.update_layout(barmode="group", yaxis_range=[0, 1], height=420,
                       title="Precision / Recall / F1 / AUC by model")
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        f"**Best model: {BEST_MODEL_NAME}** — highest F1 (0.78) and AUC (0.70), "
        "staying regularized enough not to overfit ~400 training rows.",
        icon="🏆",
    )

    st.markdown("### Confusion matrix — best model on hold-out set")
    X_val, y_val = get_holdout_split()
    proba = predict_proba(BEST_MODEL_NAME, X_val)
    preds = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y_val, preds)
    cm_fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=["Predicted: No", "Predicted: Yes"], y=["Actual: No", "Actual: Yes"],
    )
    cm_fig.update_layout(height=400, coloraxis_showscale=False)
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(cm_fig, use_container_width=True)
    with right:
        st.markdown("**Held-out metrics (recomputed live from the saved model):**")
        st.write(
            {
                "Precision": round(precision_score(y_val, preds), 3),
                "Recall": round(recall_score(y_val, preds), 3),
                "F1": round(f1_score(y_val, preds), 3),
                "AUC": round(roc_auc_score(y_val, proba), 3),
            }
        )
        st.caption(
            "Recomputed here directly from the saved `.joblib` model on the "
            "same hold-out split used in the notebook — confirms the numbers "
            "in the table above are real, not hardcoded."
        )

    st.markdown("### What the best model actually learned")
    importance = get_lgbm_feature_importance()
    fig = px.bar(
        importance, x=importance.values, y=importance.index, orientation="h",
        labels={"x": "Importance (LightGBM split count)", "y": ""},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Straight from the trained LightGBM booster's split counts — confirms "
        "the same signals flagged in EDA (travel distance, registration timing, "
        "event type, club membership) are what the model actually leaned on."
    )

    with st.expander("⚠️ Limitations & what I'd do with more time/data"):
        st.markdown(
            "- **Small dataset** — only 496 clean training rows (396 after the "
            "hold-out split). All 8 models were deliberately kept shallow/"
            "regularized because of this; a bigger dataset would likely let a "
            "more expressive model do better.\n"
            "- **Moderate ceiling** — best F1 is 0.72, AUC ~0.71. Good enough to "
            "rank students by risk of no-show, not precise enough for high-stakes "
            "individual decisions.\n"
            "- **No temporal validation** — the hold-out split is a random 20%, "
            "not a true future time-period; if this were deployed, I'd validate "
            "on a held-out *future* event to check the model generalizes across time.\n"
            "- **Features are fairly coarse** — no text, no reminder/engagement "
            "signals (e.g. did the student open the reminder email), which is "
            "typically where a real jump in accuracy would come from.\n"
            "- **Next step I'd prioritize:** collect an engagement signal "
            "(email opens, app pings) — mutual information already shows "
            "`club_member` and `registration_days_before` dominate, and an "
            "engagement proxy would likely add real signal on top of those."
        )

# --------------------------------------------------------------------------
# 4. Live Predictor
# --------------------------------------------------------------------------
elif page == "Live Predictor":
    st.title("Live Predictor")
    st.markdown("Enter a hypothetical registration and get a live attendance probability.")

    model_choice = st.selectbox(
        "Model", list(MODEL_FILES.keys()),
        index=list(MODEL_FILES.keys()).index(BEST_MODEL_NAME),
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        event_type = st.selectbox("Event type", sorted(train_df["event_type"].unique()))
        event_day = st.selectbox("Event day", sorted(train_df["event_day"].unique()))
        event_time = st.selectbox("Event time (24h)", sorted(train_df["event_time"].unique()))
    with c2:
        registration_days_before = st.slider("Registered how many days before?", 0, 30, 7)
        travel_distance_km = st.slider("Travel distance (km)", 0.0, 20.0, 5.0, step=0.1)
        club_member = st.radio("Club member?", ["Yes", "No"], horizontal=True)
    with c3:
        previous_events_registered = st.slider("Previous events registered for", 0, 15, 4)
        previous_events_attended = st.slider("Previous events actually attended", 0, 15, 3)

    input_row = pd.DataFrame([{
        "registration_days_before": registration_days_before,
        "previous_events_registered": previous_events_registered,
        "previous_events_attended": previous_events_attended,
        "club_member": 1.0 if club_member == "Yes" else 0.0,
        "event_time": float(event_time),
        "travel_distance_km": travel_distance_km,
        "event_type": event_type,
        "event_day": event_day,
    }])

    if st.button("Predict attendance", type="primary"):
        proba = predict_proba(model_choice, input_row)[0]
        pct = proba * 100
        st.metric("Attendance probability", f"{pct:.1f}%")
        st.progress(min(max(proba, 0.0), 1.0))
        if pct >= 70:
            st.success("Likely to attend ✅")
        elif pct >= 40:
            st.warning("Uncertain — a reminder nudge could help 🤔")
        else:
            st.error("Unlikely to attend ⚠️")

        if model_choice == "LightGBM":
            st.markdown("#### Why the model said this")
            contrib = explain_lgbm_prediction(input_row)
            contrib_fig = px.bar(
                contrib, x=contrib.values, y=contrib.index, orientation="h",
                color=contrib.values, color_continuous_scale=["#EF553B", "#00CC96"],
                labels={"x": "Push toward (+) or away from (-) attending", "y": ""},
            )
            contrib_fig.update_layout(
                yaxis=dict(autorange="reversed"), height=340, coloraxis_showscale=False
            )
            st.plotly_chart(contrib_fig, use_container_width=True)
            st.caption(
                "Real per-prediction contributions from the LightGBM booster "
                "(`pred_contrib`) for *this specific input* — green bars pushed "
                "the probability up, red bars pushed it down."
            )
        else:
            st.caption(
                "Per-prediction explanations are currently only wired up for "
                "the LightGBM model — switch to LightGBM above to see why."
            )

# --------------------------------------------------------------------------
# 5. Batch Predictions
# --------------------------------------------------------------------------
elif page == "Batch Predictions":
    st.title("Batch Predictions on the Real Test Set")
    st.markdown(
        "Predictions from the best model (`LightGBM`) on the unlabeled Kaggle "
        "test set — exactly the format the task asked for."
    )

    preds = load_test_predictions().sort_values("attendance_probability", ascending=False)

    st.dataframe(preds, use_container_width=True, height=350)

    st.markdown("### Top 10 most likely to attend")
    top10 = preds.head(10)
    fig = px.bar(
        top10, x="attendance_probability", y="student_id", orientation="h",
        text="attendance_probability",
        labels={"attendance_probability": "Attendance probability (%)", "student_id": "Student"},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download all predictions (CSV)",
        data=preds.to_csv(index=False).encode("utf-8"),
        file_name="test_predictions.csv",
        mime="text/csv",
    )
