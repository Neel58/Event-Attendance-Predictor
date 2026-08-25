# 🎟️ Event Attendance Predictor

Predicts whether a student who registers for a club event will actually **show up** — as a probability, e.g. *"S1475 → 94.6% likely to attend."*

`raw data → deep cleaning → EDA → 6 models trained & compared → best model deployed`

---

## The Data Was Messy — On Purpose, Cleaned Carefully

Real registration data, real problems: inconsistent labels (`"Yes"` / `"YES"` / `"yes"`), 7 duplicate rows, 3 rows where students "attended more events than they registered for," negative registration dates, and outliers up to **120 km** travel distance.

**Approach:** flag every suspicious value as missing rather than guess or drop rows → impute using values *learned only from training data* → apply identically to test. Zero missing values, zero duplicates, zero data leakage. **508 → 496 clean, usable rows.**

## What the Data Actually Says

Statistical testing (not guessing) found:

- 🏅 **Club membership is the #1 driver** — members attend 70% of the time vs. 51% for non-members
- ⏰ **Booking early predicts showing up** (p < 0.0001) — the earlier the registration, the higher the odds
- 🎯 **Event type matters** — Workshops convert at 74%, Hackathons/Socials at ~53%
- 🤷 Previous attendance history and event day/time, individually, barely move the needle

## Nine Models, Fairly Compared

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| 🏆 **GradientBoosting** | 0.79 | 0.65 | **0.71** | **0.73** |
| LightGBM | 0.79 | 0.65 | 0.71 | 0.71 |
| HistGradientBoosting | 0.74 | 0.68 | 0.71 | 0.68 |
| CatBoost | 0.76 | 0.65 | 0.70 | 0.71 |
| SVM (RBF) | 0.75 | 0.63 | 0.69 | 0.68 |
| XGBoost | 0.76 | 0.60 | 0.67 | 0.68 |
| SVM (Linear) | 0.70 | 0.63 | 0.67 | 0.66 |
| Logistic Regression | 0.73 | 0.59 | 0.65 | 0.68 |
| AdaBoost | 0.71 | 0.59 | 0.64 | 0.65 |

All 9 use class-balancing (63/37 imbalance) and deliberately heavy regularization — with only ~400 rows, an unconstrained model memorizes noise, not patterns. **GradientBoosting** ties LightGBM's F1 and wins on AUC, making it the new deployed model (`model/gradient_boosting.joblib`); **HistGradientBoosting** is a close third with the best recall of any model — worth a look if catching more true attendees matters more than precision.

**Why did LightGBM (and SVM-RBF) beat Logistic Regression?** LR can only draw a *straight line* through the data. But attendance depends on *interactions* — e.g. early registration only matters if you're a club member. LightGBM's trees capture those interactions natively; SVM-RBF gets partial credit via its kernel, which is why it also beats the linear models, just less than LightGBM.

**Did stacking all 6 models help?** No — tested it, and it made things *worse* (F1 dropped to 0.66). The 6 models correlate 0.74–0.97 with each other (too similar to be complementary), and with only ~400 rows the meta-learner overfit on noise — it actually assigned LightGBM, the *best* model, a **negative weight**. Stacking is a bet on diverse models + more data; this project had neither.

## 3 Insights for the Club

1. **Grow membership, not just registrations** — it's the single strongest predictor of showing up.
2. **Reward early sign-ups** — last-minute registrants are measurably less likely to attend.
3. **Hackathons & Socials need a nudge** (reminder / RSVP confirm) — they convert ~20 points worse than Workshops.

## Run It

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Repo:** `model/` (6 saved models) · `train_clean.csv` / `test_clean.csv` (cleaned data) · `test_predictions.csv` (final output) · `app.py` (interactive demo)

**Honest caveat:** ~500 rows means these metrics have real uncertainty — precision (0.79) is stronger than recall (0.65), so the model is more confident when it says "will attend" than it is at catching every attendee.