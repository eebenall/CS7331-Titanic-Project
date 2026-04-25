import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

train_path = "data/train.csv"

if not os.path.exists(train_path):
    raise FileNotFoundError("train.csv not found in data/train.csv")

df = pd.read_csv(train_path)

df["Title"] = df["Name"].str.split(",").str[1].str.split(".").str[0].str.strip()
df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
df["Title"] = df["Title"].replace("Mme", "Mrs")
df["Title"] = df["Title"].replace(
    ["Dr", "Rev", "Col", "Major", "Capt", "Don", "Sir", "Lady", "the Countess", "Jonkheer"],
    "Rare"
)

for title in df["Title"].unique():
    median_age = df.loc[df["Title"] == title, "Age"].median()
    df.loc[(df["Age"].isnull()) & (df["Title"] == title), "Age"] = median_age

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["IsLargeFamily"] = (df["FamilySize"] >= 5).astype(int)

def family_group(size):
    if size == 1:
        return "Alone"
    elif size <= 4:
        return "Small"
    else:
        return "Large"

df["FamilyGroup"] = df["FamilySize"].apply(family_group)

features_no_family = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title"]
features_raw_family = features_no_family + ["FamilySize"]
features_engineered_family = features_no_family + ["IsAlone", "IsLargeFamily", "FamilyGroup"]

target = "Survived"

family_rates = df.groupby("FamilyGroup")[target].mean().reindex(["Alone", "Small", "Large"])

plt.figure(figsize=(7, 5))
bars = plt.bar(family_rates.index, family_rates.values)
plt.ylim(0, 1)
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Family Group")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.1%}", ha="center")
plt.tight_layout()
plt.savefig("family_size_survival.png", dpi=150)

def run_variant(variant_name, feature_cols):
    X = df[feature_cols]
    y = df[target]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Naive Bayes": {
            "pipeline": Pipeline([
                ("preprocess", preprocessor),
                ("model", GaussianNB())
            ]),
            "params": {
                "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
            }
        },
        "KNN": {
            "pipeline": Pipeline([
                ("preprocess", preprocessor),
                ("model", KNeighborsClassifier())
            ]),
            "params": {
                "model__n_neighbors": [3, 5, 7, 9, 11, 13],
                "model__weights": ["uniform", "distance"]
            }
        }
    }

    results = []

    for model_name, config in models.items():
        grid = GridSearchCV(
            config["pipeline"],
            config["params"],
            cv=5,
            scoring="roc_auc"
        )

        grid.fit(X_train, y_train)

        predictions = grid.predict(X_test)
        probabilities = grid.predict_proba(X_test)[:, 1]

        results.append({
            "Variant": variant_name,
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "F1": f1_score(y_test, predictions),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "Best Params": grid.best_params_
        })

    return results

all_results = []
all_results.extend(run_variant("No Family Features", features_no_family))
all_results.extend(run_variant("Raw FamilySize Feature", features_raw_family))
all_results.extend(run_variant("Engineered Family Features", features_engineered_family))

results_df = pd.DataFrame(all_results)

print("\nFamily Size Survival Rates")
print(family_rates.round(3))

print("\nFinal Model Comparison")
print(results_df[["Variant", "Model", "Accuracy", "F1", "ROC-AUC"]].round(4))

results_df.to_csv("arturo_knn_nb_family_results.csv", index=False)

print("\nSaved: family_size_survival.png")
print("Saved: arturo_knn_nb_family_results.csv")
