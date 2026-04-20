import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

train_path = "data/train.csv"
if not os.path.exists(train_path):
    raise FileNotFoundError("train.csv not found in data/train.csv")

df = pd.read_csv(train_path)

df["Title"] = df["Name"].str.split(",").str[1].str.split(".").str[0].str.strip()
df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
df["Title"] = df["Title"].replace("Mme", "Mrs")
rare = ["Dr", "Rev", "Col", "Major", "Capt", "Don", "Sir", "Lady", "the Countess", "Jonkheer"]
df["Title"] = df["Title"].replace(rare, "Rare")

for t in df["Title"].unique():
    median_age = df[df["Title"] == t]["Age"].median()
    df.loc[(df["Age"].isnull()) & (df["Title"] == t), "Age"] = median_age

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

def family_group(x):
    if x == 1:
        return "Alone"
    elif x <= 4:
        return "Small"
    else:
        return "Large"

df["FamilyGroup"] = df["FamilySize"].apply(family_group)

df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])
df["Title"] = LabelEncoder().fit_transform(df["Title"])

y = df["Survived"]

features_without_family = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title"]
features_with_family = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "FamilySize"]

X_no_family = df[features_without_family]
X_with_family = df[features_with_family]

X_train_nf, X_test_nf, y_train, y_test = train_test_split(
    X_no_family, y, test_size=0.2, random_state=42, stratify=y
)

X_train_wf, X_test_wf, _, _ = train_test_split(
    X_with_family, y, test_size=0.2, random_state=42, stratify=y
)

scaler_nf = StandardScaler()
X_train_nf = scaler_nf.fit_transform(X_train_nf)
X_test_nf = scaler_nf.transform(X_test_nf)

scaler_wf = StandardScaler()
X_train_wf = scaler_wf.fit_transform(X_train_wf)
X_test_wf = scaler_wf.transform(X_test_wf)

nb_grid = GridSearchCV(
    GaussianNB(),
    {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
    cv=5,
    scoring="f1"
)
nb_grid.fit(X_train_wf, y_train)

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    {"n_neighbors": [3, 5, 7, 9, 11, 13], "weights": ["uniform", "distance"]},
    cv=5,
    scoring="f1"
)
knn_grid.fit(X_train_wf, y_train)

def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "cm": confusion_matrix(y_test, pred),
        "report": classification_report(y_test, pred)
    }

results = {}

results["Naive Bayes - No FamilySize"] = evaluate(
    GaussianNB(**nb_grid.best_params_), X_train_nf, X_test_nf, y_train, y_test
)
results["Naive Bayes - With FamilySize"] = evaluate(
    GaussianNB(**nb_grid.best_params_), X_train_wf, X_test_wf, y_train, y_test
)
results["KNN - No FamilySize"] = evaluate(
    KNeighborsClassifier(**knn_grid.best_params_), X_train_nf, X_test_nf, y_train, y_test
)
results["KNN - With FamilySize"] = evaluate(
    KNeighborsClassifier(**knn_grid.best_params_), X_train_wf, X_test_wf, y_train, y_test
)

print("\nFamily size survival rates:")
print(df.groupby("FamilyGroup")["Survived"].mean().round(3))

print("\nBest NB params:", nb_grid.best_params_)
print("Best KNN params:", knn_grid.best_params_)

print("\nModel comparison:")
for name, res in results.items():
    print(f"{name}: Accuracy={res['accuracy']:.4f}, F1={res['f1']:.4f}")

print("\nDetailed reports:")
for name, res in results.items():
    print("\n" + "=" * 60)
    print(name)
    print("Confusion Matrix:")
    print(res["cm"])
    print(res["report"])

family_survival = df.groupby("FamilyGroup")["Survived"].mean().reindex(["Alone", "Small", "Large"])
plt.figure(figsize=(7, 5))
bars = plt.bar(family_survival.index, family_survival.values)
plt.ylim(0, 1)
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Family Group")
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.1%}", ha="center")
plt.tight_layout()
plt.savefig("family_size_survival.png", dpi=150)
print("\nSaved: family_size_survival.png")
