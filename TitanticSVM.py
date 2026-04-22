"""
CS7331 Titanic Project - Support Vector Machines (SVM)
Group 5: Everett Benally, Arturo Perez Espinosa, Jeena Khatri

Improvements over v2:
  - Section 2: Data Quality Audit — prints a missing-value report before any cleaning
  - Age:    Title-group median imputation (smarter than global median)
  - Cabin:  ~77% missing → binary HasCabin flag (1 = cabin recorded, 0 = unknown)
            Deck letter extracted where available as an ordinal feature
  - Fare:   1 missing value → imputed with median of same Pclass
  - Embarked: 2 missing values → imputed with mode
  - Fare outlier capping at 99th percentile (log-transform applied inside pipeline)
  - Missing-value summary printed after cleanup to confirm no leakage
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("C:/Users/EverettBenally/Documents/eebenall/CS7331-Data_Mining/Project/titanic/train.csv")
df = df.dropna(subset=["Survived"])   # safety — target must exist

# ─────────────────────────────────────────────
# 2. DATA QUALITY AUDIT  (before any cleaning)
# ─────────────────────────────────────────────
print("=" * 55)
print("DATA QUALITY AUDIT — raw dataset")
print("=" * 55)
print(f"Total rows: {len(df)}\n")

missing = (
    df.isnull()
      .sum()
      .rename("Missing")
      .to_frame()
)
missing["Pct Missing"] = (missing["Missing"] / len(df) * 100).round(1)
missing = missing[missing["Missing"] > 0].sort_values("Missing", ascending=False)
print(missing.to_string())
print()

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING  (before imputation so
#    Title exists for age-imputation lookup)
# ─────────────────────────────────────────────

# --- Title extraction & grouping ---
df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)

title_mapping = {
    'Mr':       'Mr',
    'Miss':     'Miss',
    'Mrs':      'Mrs',
    'Master':   'Master',
    'Dr':       'Officer',
    'Rev':      'Officer',
    'Col':      'Officer',
    'Major':    'Officer',
    'Mlle':     'Miss',       # French equivalent of Miss
    'Mme':      'Mrs',        # French equivalent of Mrs
    'Countess': 'Royalty',
    'Lady':     'Royalty',
    'Sir':      'Royalty',
    'Jonkheer': 'Royalty',
    'Don':      'Royalty',
    'Dona':     'Royalty',
    'Capt':     'Officer',
}
df["Title"] = df["Title"].map(title_mapping).fillna("Other")

# --- Family size & IsAlone ---
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

# ─────────────────────────────────────────────
# 4. TARGETED MISSING-VALUE PREPROCESSING
# ─────────────────────────────────────────────

# ── 4a. AGE  (~19% missing) ──────────────────
# Strategy: impute with the median Age of passengers sharing the same
# Title group. "Master" (young boys) have a very different median than "Mr",
# so a global median would introduce systematic bias.
age_medians = df.groupby("Title")["Age"].median()
print("Age medians by Title group (used for imputation):")
print(age_medians.to_string(), "\n")

def impute_age(row):
    if pd.isnull(row["Age"]):
        return age_medians.get(row["Title"], df["Age"].median())
    return row["Age"]

df["Age"] = df.apply(impute_age, axis=1)

# ── 4b. CABIN  (~77% missing) ────────────────
# Strategy: The cabin number itself is nearly unusable, but *having* a cabin
# recorded is a strong proxy for 1st-class wealth.
# (a) Binary flag: 1 = cabin was recorded, 0 = unknown
df["HasCabin"] = df["Cabin"].notna().astype(int)

# (b) Deck letter: extract the first character where available,
#     assign "U" (Unknown) otherwise. Deck maps loosely to ship level,
#     which correlates with both Pclass and proximity to lifeboats.
df["Deck"] = df["Cabin"].str[0].fillna("U")

# Collapse rare decks (fewer than 10 passengers) into "Other"
deck_counts = df["Deck"].value_counts()
rare_decks  = deck_counts[deck_counts < 10].index
df["Deck"]  = df["Deck"].replace(rare_decks, "Other")
print("Deck distribution after collapsing rare decks:")
print(df["Deck"].value_counts().to_string(), "\n")

# ── 4c. FARE  (1 missing) ────────────────────
# Strategy: impute with median Fare of passengers in the same Pclass.
# A global median would be wrong — 3rd-class fares are much lower than 1st.
fare_medians = df.groupby("Pclass")["Fare"].median()
df["Fare"] = df.apply(
    lambda r: fare_medians[r["Pclass"]] if pd.isnull(r["Fare"]) else r["Fare"],
    axis=1
)

# ── 4d. EMBARKED  (2 missing) ────────────────
# Strategy: fill with mode (S = Southampton, the most common port).
embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)
print(f"Embarked missing values filled with mode: '{embarked_mode}'\n")

# ── 4e. FARE OUTLIER CAPPING ─────────────────
# A handful of 1st-class fares are extreme outliers (e.g. £512).
# Cap at 99th percentile before log-transforming to compress the distribution.
fare_cap = df["Fare"].quantile(0.99)
df["Fare"] = df["Fare"].clip(upper=fare_cap)
print(f"Fare capped at 99th percentile: £{fare_cap:.2f}\n")

# ─────────────────────────────────────────────
# 5. POST-CLEANUP AUDIT
# ─────────────────────────────────────────────
print("=" * 55)
print("DATA QUALITY AUDIT — after preprocessing")
print("=" * 55)
remaining = df[["Age", "Cabin", "Fare", "Embarked", "HasCabin", "Deck"]].isnull().sum()
print(remaining.to_string())
print("(Cabin column retained in df but NOT used as a model feature)\n")

# ─────────────────────────────────────────────
# 6. FEATURE SELECTION
# ─────────────────────────────────────────────
# Numeric  : Pclass kept as ordinal int; log1p(Fare) applied inside pipeline
# Binary   : HasCabin, IsAlone — already 0/1, treated as numeric
# Categorical: Sex, Embarked, Title, Deck — one-hot encoded
num_features = ["Pclass", "Age", "Fare", "FamilySize", "HasCabin", "IsAlone"]
cat_features = ["Sex", "Embarked", "Title", "Deck"]

features = num_features + cat_features
target   = "Survived"

X = df[features].copy()
y = df[target].astype(int)

# ─────────────────────────────────────────────
# 7. PREPROCESSING PIPELINES
# ─────────────────────────────────────────────
# Fare gets a log1p transform BEFORE scaling to reduce right-skew.
# Split numerics into two sub-pipelines so only Fare is log-transformed.

log_transform = FunctionTransformer(np.log1p, validate=False)

fare_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # safety net only
    ("log",     log_transform),
    ("scaler",  StandardScaler())
])

plain_num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

plain_num_features = [f for f in num_features if f != "Fare"]

preprocessor = ColumnTransformer([
    ("fare", fare_pipeline,      ["Fare"]),
    ("num",  plain_num_pipeline, plain_num_features),
    ("cat",  cat_pipeline,       cat_features)
])

svm_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier",    SVC())
])

# ─────────────────────────────────────────────
# 8. STRATIFIED TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Train survival rate: {y_train.mean():.3f} | "
      f"Test survival rate: {y_test.mean():.3f}\n")

# ─────────────────────────────────────────────
# 9. EXPANDED HYPERPARAMETER GRID
# ─────────────────────────────────────────────
param_grid = [
    {   # Linear kernel
        "classifier__kernel":       ["linear"],
        "classifier__C":            [0.01, 1, 10, 100],
        "classifier__class_weight": ["balanced"]
    },
    {   # RBF kernel
        "classifier__kernel":       ["rbf"],
        "classifier__C":            [1, 5, 10],
        "classifier__gamma":        ["scale", 0.05, 0.1],
        "classifier__class_weight": ["balanced"]
    },
    {   # Polynomial kernel
        "classifier__kernel":       ["poly"],
        "classifier__C":            [0.1, 1, 10],
        "classifier__degree":       [2],
        "classifier__class_weight": ["balanced"]
    }
]

grid = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

print("Running GridSearchCV — this may take a minute...")
grid.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 10. EVALUATION
# ─────────────────────────────────────────────
y_pred = grid.predict(X_test)

best_cv_score = grid.best_score_
test_accuracy = accuracy_score(y_test, y_pred)
gap           = test_accuracy - best_cv_score

print("\n" + "=" * 55)
print("RESULTS")
print("=" * 55)
print(f"Best Params:       {grid.best_params_}")
print(f"Best CV Accuracy:  {best_cv_score:.4f}")
print(f"Test Accuracy:     {test_accuracy:.4f}")
print(f"CV → Test Gap:     {gap:+.4f}  "
      f"{'(potential overfit)' if gap < -0.02 else '(looks healthy)'}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

# ─────────────────────────────────────────────
# 11. MISSING VALUE IMPACT SUMMARY PLOT
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
raw_missing_pct = {
    "Age":       19.9,
    "Cabin":     77.1,
    "Embarked":   0.2,
    "Fare":       0.1,
}
strategy_labels = {
    "Age":      "Title-group median imputation",
    "Cabin":    "Binary flag (HasCabin) + Deck extraction",
    "Embarked": "Mode fill (S = Southampton)",
    "Fare":     "Pclass-group median + 99th-pct cap",
}
bar_colors = ["#e07b54", "#c0392b", "#7fb3d3", "#a9cce3"]
bars = ax.barh(
    list(raw_missing_pct.keys()),
    list(raw_missing_pct.values()),
    color=bar_colors,
    edgecolor="white"
)
for bar, col in zip(bars, raw_missing_pct.keys()):
    ax.text(
        bar.get_width() + 0.8,
        bar.get_y() + bar.get_height() / 2,
        strategy_labels[col],
        va="center", fontsize=8.5, color="#333333"
    )
ax.set_xlabel("% Missing (raw dataset)", fontsize=10)
ax.set_title("Missing Value Summary & Imputation Strategies", fontsize=11)
ax.set_xlim(0, 115)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("svm_missing_value_summary.png", dpi=150)
print("Saved: svm_missing_value_summary.png")

# ─────────────────────────────────────────────
# 12. CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Died", "Survived"],
    colorbar=False,
    ax=ax
)
ax.set_title(
    f"SVM Confusion Matrix\n"
    f"kernel={grid.best_params_['classifier__kernel']}  "
    f"C={grid.best_params_['classifier__C']}  "
    f"Acc={test_accuracy:.3f}"
)
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png", dpi=150)
print("Saved: svm_confusion_matrix.png")

# ─────────────────────────────────────────────
# 13. PCA DECISION BOUNDARY VISUALIZATION
# ─────────────────────────────────────────────
print("\nGenerating PCA decision boundary visualization...")

best_preprocessor   = grid.best_estimator_.named_steps["preprocessing"]
X_train_transformed = best_preprocessor.transform(X_train)

pca        = PCA(n_components=2, random_state=42)
X_2d_train = pca.fit_transform(X_train_transformed)

best_params = {
    k.replace("classifier__", ""): v
    for k, v in grid.best_params_.items()
}
best_params.pop("probability", None)

vis_svm = SVC(**best_params)
vis_svm.fit(X_2d_train, y_train)

x_min, x_max = X_2d_train[:, 0].min() - 1, X_2d_train[:, 0].max() + 1
y_min, y_max = X_2d_train[:, 1].min() - 1, X_2d_train[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)
Z = vis_svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
ax.contour(xx, yy, Z, colors="k", linewidths=0.8, linestyles="--")

colors = {0: "tomato", 1: "steelblue"}
labels = {0: "Died",   1: "Survived"}
for cls in [0, 1]:
    mask = (y_train == cls).values
    ax.scatter(
        X_2d_train[mask, 0], X_2d_train[mask, 1],
        c=colors[cls], label=labels[cls],
        edgecolors="k", linewidths=0.4, s=30, alpha=0.7
    )

pca_var = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1 ({pca_var[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca_var[1]*100:.1f}% variance)")
ax.set_title(
    f"SVM Decision Boundary (PCA 2D Projection)\n"
    f"kernel={grid.best_params_['classifier__kernel']}  "
    f"C={grid.best_params_['classifier__C']}"
)
ax.legend()
plt.tight_layout()
plt.savefig("svm_decision_boundary.png", dpi=150)
print("Saved: svm_decision_boundary.png")

print("\nDone.")