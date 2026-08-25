# 🎟️ Event Attendance Predictor

Predicts whether a student who registered for a club event will actually **show up** — output as a probability like *"87% likely to attend"* — so the club can plan capacity, follow-ups, and reminders around who's actually likely to come.

---

## 1. The Problem

Clubs collect registrations, but registrations ≠ attendance. This project turns raw, messy registration data into a probability score per student, using six candidate models, and explains **which** patterns actually drive attendance.

**Pipeline:** `raw data → deep cleaning → EDA → 6 trained models → best model selected on F1 → probability predictions`

| Notebook | Purpose |
|---|---|
| `event-attendance-predictor-cleanup-notebook.ipynb` | Fixes data quality issues, produces `train_clean.csv` / `test_clean.csv` |
| `event-attendance-predictor-deep-eda.ipynb` | Statistically validates which features actually matter |
| `event-attendance-predictor-modeling.ipynb` | Trains, evaluates, and saves 6 classifiers |

---

## 2. The Dataset

508 training registrations / 100 test registrations, with fields like `event_type`, `registration_days_before`, `previous_events_registered`, `previous_events_attended`, `club_member`, `event_day`, `event_time`, `travel_distance_km`, and the label `attended`.

## 3. Why So Much Cleaning? What Was Actually Wrong

The raw data wasn't just "a few blanks" — it had real, structural quality problems. A quality audit (missing counts, duplicate checks, category value inspection, logical-consistency checks) surfaced:

- **Missing values everywhere** — every single column had gaps (up to 15/508 rows for some fields).
- **Inconsistent category spelling** — `event_type` had `"Hackathon"`, `"HACKATHON"`, and `"hackathon"` as three different strings; `club_member` had `"Yes"`, `"YES"`, `"yes"`. Left as-is, a model would treat these as different categories.
- **Duplicate records** — 7 exact duplicate rows, plus repeated `student_id`s.
- **Logically impossible values** — 3 rows where a student had *attended more events than they'd registered for*, and 2 rows with a *negative* `registration_days_before` (registering before the event was announced).
- **Extreme outliers** — e.g. a `registration_days_before` of 60 and a `travel_distance_km` of 120, far outside the rest of the distribution.

**The approach:** rather than dropping rows (and losing already-scarce data) or guessing at "correct" values, every suspicious value was converted to `NaN` and then **imputed** (median for numeric columns, mode for categorical). This keeps every usable row while refusing to let broken data quietly bias the model. Impossible-value rows in particular were fully nulled rather than "fixed," because there's no way to know which of the two conflicting numbers was actually the error — inventing a swap would just introduce a different kind of bias.

Critically, the cleaning logic is a `DataCleaner` class that **fits only on train** (learns outlier bounds and fill values from training data) and then **applies those exact learned values to the test set** — never re-computing statistics from test data. This avoids data leakage and mirrors how the model would behave on genuinely new registrations in production. Rows with a missing target label were kept aside in `train_unlabeled.csv` instead of being deleted, since they still had usable feature information.

Net effect: 508 → 496 usable labeled training rows, 0 missing values, 0 duplicates, 0 impossible values — the same transformation applied identically to the 100 test rows.

## 4. EDA — What the Data Actually Says

Before modeling, every feature was statistically tested against the target (point-biserial correlation, Mann-Whitney U for numeric features; chi-square + Cramér's V for categorical features; mutual information for overall ranking) instead of guessing which features matter.

**Headline findings:**

- **Class balance:** 63% attended vs. 37% didn't — imbalanced enough that every model needed class-weighting, but not severe enough to need resampling.
- **Club membership is the strongest signal** (highest mutual information, statistically significant): members attend 70% of the time vs. 51% for non-members.
- **Booking early matters:** `registration_days_before` is the second-strongest signal and the only numeric feature with a clear, statistically significant relationship (p < 0.0001) — students who register further ahead attend more.
- **Event type matters, workshops win:** attendance ranges from 53% (Hackathon/Social) up to 74% (Workshop) — a significant chi-square relationship.
- **Several features carry almost no signal:** `previous_events_registered`, `previous_events_attended`, `event_time`, and `event_day` all showed mutual information ≈ 0 and non-significant p-values individually. They weren't dropped (models can still exploit interactions and non-linear combinations LightGBM/CatBoost pick up automatically), but they don't explain attendance on their own.
- **No train/test drift:** every feature passed a Kolmogorov–Smirnov / chi-square distribution-shift test with p > 0.4, confirming the test set is drawn from the same population — so a model validated on the hold-out split should generalize to the real test set.

## 5. Modeling — 6 Classifiers, Evaluated Fairly

Six classifiers were trained on a stratified 80/20 hold-out split of the cleaned training data: **Logistic Regression, SVM (Linear), SVM (RBF), XGBoost, LightGBM, CatBoost.**

Two design decisions applied to *every* model, given how little data there is (~400 training rows):
- **Class-weight balancing** (`class_weight="balanced"` / `scale_pos_weight` / `auto_class_weights`) instead of letting models default to predicting the majority class.
- **Deliberately strong regularization** — shallow trees, low leaf/child counts, L1/L2 penalties, subsampling. With this few rows, an unconstrained model would memorize noise rather than learn a real pattern, so every model was tuned to stay simple on purpose.

### Results (hold-out set, 100 rows)

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **LightGBM** | 0.7885 | 0.6508 | **0.7130** | 0.7113 |
| CatBoost | 0.7593 | 0.6508 | 0.7009 | 0.7134 |
| SVM (RBF) | 0.7547 | 0.6349 | 0.6897 | 0.6821 |
| XGBoost | 0.7600 | 0.6032 | 0.6726 | 0.6778 |
| SVM (Linear) | 0.7018 | 0.6349 | 0.6667 | 0.6572 |
| Logistic Regression | 0.7255 | 0.5873 | 0.6491 | 0.6847 |

**LightGBM was selected as the final model** (best F1) and used to score the real, unlabeled test set — output saved as `test_predictions.csv` (student ID → attendance probability, e.g. `S1475 → 94.6% likely to attend`).

### Why did LightGBM (and SVM-RBF) beat plain Logistic Regression?

Logistic Regression can only draw a single **straight-line (linear) boundary** through the feature space — it assumes attendance probability changes in one consistent direction as each feature increases. The EDA above shows that isn't how this data behaves: e.g. `event_type × club_member × event_time` interact (a member at a 9am Workshop behaves very differently from a non-member at the same slot), and several features (`event_time`, `event_day`) only matter *in combination* with others, not on their own.

- **LightGBM** builds decision trees that split on thresholds and automatically capture these feature *interactions* and non-linear effects (e.g. "early registration only matters if you're a club member") without anyone hand-engineering them. It also handles the categorical columns as native splits rather than diluting the signal into many sparse one-hot columns.
- **SVM (RBF)** used a Gaussian kernel to project the data into a higher-dimensional space, letting it draw a *curved* decision boundary instead of a straight line — which is why it also beat both the linear SVM and plain Logistic Regression, just not by as much as LightGBM, since it doesn't get LightGBM's built-in categorical-interaction handling.
- **Plain Logistic Regression and Linear SVM**, being linear-only, could only "average out" these interaction effects, which is why they land at the bottom of the F1 ranking.

CatBoost is close behind LightGBM (best AUC of all models, second-best F1) for the same structural reason — it's also a tree-based, interaction-aware model, just with slightly less aggressive tuning here.

## 6. Actionable Insights for the Club

1. **Club membership drives attendance far more than anything else measurable here (70% vs 51%).** Membership drives (even informal ones) may do more for turnout than tweaking event logistics.
2. **Earlier registration predicts attendance.** Consider a small nudge — an early-bird perk or reminder — to pull registrations earlier, since last-minute sign-ups are measurably less likely to convert into attendance.
3. **Workshops convert best (74%); Hackathons and Socials convert worst (~53%).** For these lower-converting formats, targeted reminders or a lightweight RSVP-confirmation step could recover some of that gap. `event_day`/`event_time` alone don't move the needle much — the *type* of event is what matters, so scheduling tweaks are unlikely to help as much as changing the event format or its framing would.

## 7. Repository Structure

```
model/                          # 6 saved models (.joblib)
train_clean.csv, test_clean.csv # cleaned data (cleanup notebook output)
train_unlabeled.csv             # rows with no attendance label, kept for reference
test_predictions.csv            # final LightGBM probability predictions
app.py                          # Streamlit demo comparing all 6 models interactively
requirements.txt
```

## 8. Running It

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 9. Honest Limitations

The dataset is small (~500 rows), so metrics carry real uncertainty — F1 differences between LightGBM, CatBoost, and SVM-RBF (0.70–0.71) are close enough that any of the three would be a reasonable production choice, and the ranking could shift with more data. Precision (~0.79) is stronger than recall (~0.65) for the winning model, meaning it's more confident when it predicts "will attend" than it is at catching every student who will actually show up — worth knowing if the club's use case cares more about one error type than the other.
