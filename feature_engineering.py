from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


warnings.filterwarnings("ignore")

DATA_PATH = Path("data") / "train.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_NAMES = ["Naive Bayes", "KNN", "SVM", "Random Forest"]


def load_data(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            "train.csv not found. Please place the Titanic train file in data/train.csv."
        )

    df = pd.read_csv(path)
    print(f"Using data file: {path}")
    print("Shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    return df


def extract_titles(df):
    df = df.copy()
    df["Title"] = df["Name"].str.split(",").str[1].str.split(".").str[0].str.strip()

    print("\nTitles found:")
    print(df["Title"].value_counts())

    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    rare_titles = [
        "Dr",
        "Rev",
        "Col",
        "Major",
        "Capt",
        "Don",
        "Sir",
        "Lady",
        "the Countess",
        "Jonkheer",
    ]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")

    print("\nAfter grouping:")
    print(df["Title"].value_counts())
    print("\nSurvival rate by title:")
    print(df.groupby("Title")["Survived"].mean().round(3))
    return df


def clean_data(df):
    df = df.copy()

    for title in df["Title"].unique():
        median_age = df.loc[df["Title"] == title, "Age"].median()
        age_missing = df["Age"].isnull() & (df["Title"] == title)
        df.loc[age_missing, "Age"] = median_age

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

    title_survival = df.groupby("Title")["Survived"].mean().sort_values(ascending=False)
    df = pd.get_dummies(df, columns=["Title", "Embarked"], drop_first=False)

    print("\nCleaned data shape:", df.shape)
    print("\nNo more missing values:")
    print(df.isnull().sum())
    return df, title_survival


def build_feature_sets(df):
    y = df["Survived"]
    title_cols = [col for col in df.columns if col.startswith("Title_")]

    feature_sets = {
        "base": df[["Age", "Sex"]],
        "socio": df[["Age", "Sex", "Pclass", "Fare"] + title_cols],
        "all": df.drop(["Survived"], axis=1),
    }
    return feature_sets, y


def split_feature_sets(feature_sets, y):
    splits = {}
    y_train = None
    y_test = None

    for name, X in feature_sets.items():
        X_train, X_test, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        splits[name] = (X_train, X_test)

        if y_train is None:
            y_train, y_test = y_tr, y_te

    return splits, y_train, y_test


def make_scaled_pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def tune_models(X_train, y_train):
    print("\nTuning hyperparameters (this takes a sec)...")

    grids = {
        "Naive Bayes": GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("nb", GaussianNB())]),
            {"nb__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
            cv=5,
            scoring="f1",
        ),
        "KNN": GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
            {
                "knn__n_neighbors": [3, 5, 7, 9, 11, 13],
                "knn__weights": ["uniform", "distance"],
            },
            cv=5,
            scoring="f1",
        ),
        "SVM": GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("svm", SVC())]),
            {"svm__C": [0.1, 1, 10], "svm__kernel": ["rbf", "linear"]},
            cv=5,
            scoring="f1",
        ),
        "Random Forest": GridSearchCV(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("rf", RandomForestClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "rf__n_estimators": [50, 100, 200],
                "rf__max_depth": [5, 10, 15, None],
            },
            cv=5,
            scoring="f1",
        ),
    }

    best_params = {}
    prefixes = {
        "Naive Bayes": "nb__",
        "KNN": "knn__",
        "SVM": "svm__",
        "Random Forest": "rf__",
    }

    for name, grid in grids.items():
        grid.fit(X_train, y_train)
        prefix = prefixes[name]
        params = {key.replace(prefix, ""): val for key, val in grid.best_params_.items()}
        best_params[name] = params
        print(f"{name} best params: {params}")

    return best_params


def get_model(name, best_params):
    if name == "Naive Bayes":
        return GaussianNB(**best_params[name])
    if name == "KNN":
        return KNeighborsClassifier(**best_params[name])
    if name == "SVM":
        return SVC(**best_params[name])
    return RandomForestClassifier(**best_params[name], random_state=RANDOM_STATE)


def evaluate_model(model, X_train, X_test, y_train, y_test):
    pipe = make_scaled_pipeline(model)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return {
        "acc": round(accuracy_score(y_test, pred), 4),
        "prec": round(precision_score(y_test, pred, zero_division=0), 4),
        "rec": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "pred": pred,
    }


def run_model_comparison(splits, y_train, y_test, best_params):
    results = []

    for name in MODEL_NAMES:
        base = evaluate_model(
            get_model(name, best_params), *splits["base"], y_train, y_test
        )
        socio = evaluate_model(
            get_model(name, best_params), *splits["socio"], y_train, y_test
        )
        full = evaluate_model(
            get_model(name, best_params), *splits["all"], y_train, y_test
        )

        results.append(
            {
                "Model": name,
                "Base Acc": base["acc"],
                "Socio Acc": socio["acc"],
                "All Acc": full["acc"],
                "Base F1": base["f1"],
                "Socio F1": socio["f1"],
                "All F1": full["f1"],
                "Socio Prec": socio["prec"],
                "Socio Rec": socio["rec"],
                "socio_pred": socio["pred"],
            }
        )

    return results


def print_results(results, y_test):
    display_cols = [
        "Model",
        "Base Acc",
        "Socio Acc",
        "All Acc",
        "Base F1",
        "Socio F1",
        "All F1",
        "Socio Prec",
        "Socio Rec",
    ]

    results_df = pd.DataFrame(results)[display_cols]
    print("\n" + "=" * 80)
    print("TUNED MODEL COMPARISON")
    print("Base = Age+Sex | Socio = Age+Sex+Pclass+Fare+Title | All = all features")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print("\n--- Accuracy gain: Base -> Socioeconomic ---")
    for result in results:
        diff = result["Socio Acc"] - result["Base Acc"]
        sign = "+" if diff >= 0 else ""
        print(f"  {result['Model']}: {sign}{diff:.4f} ({sign}{diff * 100:.2f}%)")

    print("\n--- F1 gain: Base -> Socioeconomic ---")
    for result in results:
        diff = result["Socio F1"] - result["Base F1"]
        sign = "+" if diff >= 0 else ""
        print(f"  {result['Model']}: {sign}{diff:.4f} ({sign}{diff * 100:.2f}%)")

    print("\n" + "=" * 70)
    print("DETAILED EVALUATION (Socioeconomic Feature Set)")
    print("=" * 70)

    for result in results:
        print(f"\n--- {result['Model']} ---")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, result["socio_pred"]))
        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                result["socio_pred"],
                target_names=["Did not survive", "Survived"],
            )
        )


def print_rf_overfit_check(splits, y_train, results, best_params):
    rf_pipe = make_scaled_pipeline(get_model("Random Forest", best_params))
    X_socio_train, _ = splits["socio"]
    rf_pipe.fit(X_socio_train, y_train)

    train_acc = accuracy_score(y_train, rf_pipe.predict(X_socio_train))
    test_acc = next(r for r in results if r["Model"] == "Random Forest")["Socio Acc"]
    print(f"\nRF train accuracy: {train_acc:.4f} | test accuracy: {test_acc:.4f}")


def save_title_plot(title_survival):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        title_survival.index,
        title_survival.values,
        color=["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"],
    )
    ax.set_ylabel("Survival Rate")
    ax.set_title("Survival Rate by Title")
    ax.set_ylim(0, 1)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.1%}",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig("survival_by_title.png", dpi=150)
    plt.close()
    print("\nSaved: survival_by_title.png")


def save_metric_plot(results, metric, ylabel, title, filename, color):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [r["Model"] for r in results]
    base_values = [r[f"Base {metric}"] for r in results]
    socio_values = [r[f"Socio {metric}"] for r in results]

    x = np.arange(len(models))
    width = 0.35
    base_bars = ax.bar(
        x - width / 2,
        base_values,
        width,
        label="Base (Age + Sex)",
        color="#95a5a6",
    )
    socio_bars = ax.bar(
        x + width / 2,
        socio_values,
        width,
        label="+ Socioeconomic",
        color=color,
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.6 if metric == "F1" else 0.7, 0.9 if metric == "F1" else 0.95)
    ax.legend()

    for bar in list(base_bars) + list(socio_bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f"{height:.1%}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def save_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, result in enumerate(results):
        cm = confusion_matrix(y_test, result["socio_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
        )
        axes[i].set_title(result["Model"])
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.suptitle("Confusion Matrices (Socioeconomic Features)", y=1.02)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: confusion_matrices.png")


def save_plots(results, title_survival, y_test):
    save_title_plot(title_survival)
    save_metric_plot(
        results,
        "Acc",
        "Accuracy",
        "Model Accuracy: Demographics vs Socioeconomic Features",
        "accuracy_comparison.png",
        "#2ecc71",
    )
    save_metric_plot(
        results,
        "F1",
        "F1 Score",
        "Model F1 Score: Demographics vs Socioeconomic Features",
        "f1_comparison.png",
        "#3498db",
    )
    save_confusion_matrices(results, y_test)


def main():
    raw_df = load_data()
    titled_df = extract_titles(raw_df)
    clean_df, title_survival = clean_data(titled_df)

    feature_sets, y = build_feature_sets(clean_df)
    splits, y_train, y_test = split_feature_sets(feature_sets, y)

    X_socio_train, _ = splits["socio"]
    best_params = tune_models(X_socio_train, y_train)

    results = run_model_comparison(splits, y_train, y_test, best_params)
    print_rf_overfit_check(splits, y_train, results, best_params)
    print_results(results, y_test)
    save_plots(results, title_survival, y_test)


if __name__ == "__main__":
    main()
