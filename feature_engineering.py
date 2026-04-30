import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# load dataset from the repo data folder
train_path = 'data/train.csv'
if not os.path.exists(train_path):
    raise FileNotFoundError(
        'train.csv not found. Please place the Titanic train file in data/train.csv.'
    )

df = pd.read_csv(train_path)
print(f"Using data file: {train_path}")

print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# --- STEP 1: extract title from name ---
# name format: "LastName, Title. FirstName"
# split at comma, take the part after it, then split at period to get title
df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()

print("\nTitles found:")
print(df['Title'].value_counts())

# --- STEP 2: group rare titles ---
# mlle and ms are basically miss, mme is mrs
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# everything else thats not mr/mrs/miss/master is rare
rare = ['Dr', 'Rev', 'Col', 'Major', 'Capt', 'Don', 'Sir',
        'Lady', 'the Countess', 'Jonkheer']
df['Title'] = df['Title'].replace(rare, 'Rare')

print("\nAfter grouping:")
print(df['Title'].value_counts())

# check survival rate by title
print("\nSurvival rate by title:")
print(df.groupby('Title')['Survived'].mean().round(3))

# --- STEP 3: preprocess the data ---
# fill missing age with median age per title (smarter than overall median)
for t in df['Title'].unique():
    median_age = df[df['Title'] == t]['Age'].median()
    df.loc[(df['Age'].isnull()) & (df['Title'] == t), 'Age'] = median_age

# fill missing embarked with most common value
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# drop columns we dont need
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# encode sex: male=1, female=0 (binary — no ordinal issue)
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# one-hot encode Title and Embarked (nominal — no fake ordering)
# save title survival rates before one-hot so we can use in plot later
title_surv = df.groupby('Title')['Survived'].mean().sort_values(ascending=False)
df = pd.get_dummies(df, columns=['Title', 'Embarked'], drop_first=False)

print("\nCleaned data shape:", df.shape)
print("\nNo more missing values:")
print(df.isnull().sum())

# --- STEP 4: setup for model comparison ---
y = df['Survived']

title_cols = [c for c in df.columns if c.startswith('Title_')]

# baseline: just age and sex (pure demographics)
X_base = df[['Age', 'Sex']]

# with socioeconomic: add Pclass, Fare, and one-hot Title
X_socio = df[['Age', 'Sex', 'Pclass', 'Fare'] + title_cols]

# all features for reference
X_all = df.drop(['Survived'], axis=1)

# stratified split so class balance is preserved in both sets
X_base_tr, X_base_te, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42, stratify=y)

X_socio_tr, X_socio_te, _, _ = train_test_split(
    X_socio, y, test_size=0.2, random_state=42, stratify=y)

X_all_tr, X_all_te, _, _ = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y)

# --- STEP 5: helper to train, predict and evaluate ---
# Pipeline handles scaling inside so scaler never sees test data before fit
def run_model(model, X_tr, X_te, y_tr, y_te):
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    return {
        'acc': round(accuracy_score(y_te, pred), 4),
        'prec': round(precision_score(y_te, pred, zero_division=0), 4),
        'rec': round(recall_score(y_te, pred, zero_division=0), 4),
        'f1': round(f1_score(y_te, pred, zero_division=0), 4),
        'pred': pred
    }

# --- STEP 6: hyperparameter tuning with GridSearchCV ---
# Pipelines inside GridSearchCV so scaler refits on each fold — no leakage
print("\nTuning hyperparameters (this takes a sec)...")

nb_grid = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('nb', GaussianNB())]),
    {'nb__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
    cv=5, scoring='f1')
nb_grid.fit(X_socio_tr, y_train)
nb_best = {k.replace('nb__', ''): v for k, v in nb_grid.best_params_.items()}
print(f"NB best params: {nb_best}")

knn_grid = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]),
    {'knn__n_neighbors': [3, 5, 7, 9, 11, 13], 'knn__weights': ['uniform', 'distance']},
    cv=5, scoring='f1')
knn_grid.fit(X_socio_tr, y_train)
knn_best = {k.replace('knn__', ''): v for k, v in knn_grid.best_params_.items()}
print(f"KNN best params: {knn_best}")

svm_grid = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('svm', SVC())]),
    {'svm__C': [0.1, 1, 10], 'svm__kernel': ['rbf', 'linear']},
    cv=5, scoring='f1')
svm_grid.fit(X_socio_tr, y_train)
svm_best = {k.replace('svm__', ''): v for k, v in svm_grid.best_params_.items()}
print(f"SVM best params: {svm_best}")

rf_grid = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(random_state=42))]),
    {'rf__n_estimators': [50, 100, 200], 'rf__max_depth': [5, 10, 15, None]},
    cv=5, scoring='f1')
rf_grid.fit(X_socio_tr, y_train)
rf_best = {k.replace('rf__', ''): v for k, v in rf_grid.best_params_.items()}
print(f"RF best params: {rf_best}")

# --- STEP 7: run tuned models on all 3 feature sets ---
# fresh model instance each time so no state carries over between feature sets
def get_tuned(name):
    if name == 'Naive Bayes':
        return GaussianNB(**nb_best)
    elif name == 'KNN':
        return KNeighborsClassifier(**knn_best)
    elif name == 'SVM':
        return SVC(**svm_best)
    else:
        return RandomForestClassifier(**rf_best, random_state=42)

model_names = ['Naive Bayes', 'KNN', 'SVM', 'Random Forest']
results = []

for name in model_names:
    base = run_model(get_tuned(name), X_base_tr, X_base_te, y_train, y_test)
    socio = run_model(get_tuned(name), X_socio_tr, X_socio_te, y_train, y_test)
    full = run_model(get_tuned(name), X_all_tr, X_all_te, y_train, y_test)

    results.append({
        'Model': name,
        'Base Acc': base['acc'], 'Socio Acc': socio['acc'], 'All Acc': full['acc'],
        'Base F1': base['f1'], 'Socio F1': socio['f1'], 'All F1': full['f1'],
        'Socio Prec': socio['prec'], 'Socio Rec': socio['rec'],
        'socio_pred': socio['pred']
    })

# RF overfitting check — compare train vs test accuracy
rf_pipe_check = Pipeline([('scaler', StandardScaler()), ('model', get_tuned('Random Forest'))])
rf_pipe_check.fit(X_socio_tr, y_train)
rf_train_acc = accuracy_score(y_train, rf_pipe_check.predict(X_socio_tr))
rf_test_acc = [r for r in results if r['Model'] == 'Random Forest'][0]['Socio Acc']
print(f"\nRF train accuracy: {rf_train_acc:.4f} | test accuracy: {rf_test_acc:.4f}")

# --- STEP 8: show results ---
disp_cols = ['Model', 'Base Acc', 'Socio Acc', 'All Acc', 'Base F1', 'Socio F1', 'All F1', 'Socio Prec', 'Socio Rec']
results_df = pd.DataFrame(results)[disp_cols]
print("\n" + "="*80)
print("TUNED MODEL COMPARISON")
print("Base = Age+Sex | Socio = Age+Sex+Pclass+Fare+Title | All = all features")
print("="*80)
print(results_df.to_string(index=False))

print("\n--- Accuracy gain: Base -> Socioeconomic ---")
for r in results:
    diff = r['Socio Acc'] - r['Base Acc']
    sign = '+' if diff >= 0 else ''
    print(f"  {r['Model']}: {sign}{diff:.4f} ({sign}{diff*100:.2f}%)")

print("\n--- F1 gain: Base -> Socioeconomic ---")
for r in results:
    diff = r['Socio F1'] - r['Base F1']
    sign = '+' if diff >= 0 else ''
    print(f"  {r['Model']}: {sign}{diff:.4f} ({sign}{diff*100:.2f}%)")

# --- STEP 9: detailed evaluation for socioeconomic feature set ---
print("\n" + "="*70)
print("DETAILED EVALUATION (Socioeconomic Feature Set)")
print("="*70)

for r in results:
    print(f"\n--- {r['Model']} ---")
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, r['socio_pred'])
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, r['socio_pred'],
          target_names=['Did not survive', 'Survived']))

# --- STEP 10: visualizations ---
import matplotlib.pyplot as plt
import seaborn as sns

# plot 1: survival rate by title (use title_surv saved before one-hot encoding)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(title_surv.index, title_surv.values, color=['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6'])
ax.set_ylabel('Survival Rate')
ax.set_title('Survival Rate by Title')
ax.set_ylim(0, 1)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.1%}', ha='center')
plt.tight_layout()
plt.savefig('survival_by_title.png', dpi=150)
plt.close()
print("\nSaved: survival_by_title.png")

# plot 2: accuracy comparison - base vs socioeconomic
fig, ax = plt.subplots(figsize=(10, 6))
models_names = [r['Model'] for r in results]
base_acc = [r['Base Acc'] for r in results]
socio_acc = [r['Socio Acc'] for r in results]

x = np.arange(len(models_names))
w = 0.35
b1 = ax.bar(x - w/2, base_acc, w, label='Base (Age + Sex)', color='#95a5a6')
b2 = ax.bar(x + w/2, socio_acc, w, label='+ Socioeconomic', color='#2ecc71')

ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy: Demographics vs Socioeconomic Features')
ax.set_xticks(x)
ax.set_xticklabels(models_names)
ax.set_ylim(0.7, 0.95)
ax.legend()

for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.1%}', ha='center', fontsize=9)
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.1%}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150)
plt.close()
print("Saved: accuracy_comparison.png")

# plot 3: f1 comparison
fig, ax = plt.subplots(figsize=(10, 6))
base_f1 = [r['Base F1'] for r in results]
socio_f1 = [r['Socio F1'] for r in results]

b1 = ax.bar(x - w/2, base_f1, w, label='Base (Age + Sex)', color='#95a5a6')
b2 = ax.bar(x + w/2, socio_f1, w, label='+ Socioeconomic', color='#3498db')

ax.set_ylabel('F1 Score')
ax.set_title('Model F1 Score: Demographics vs Socioeconomic Features')
ax.set_xticks(x)
ax.set_xticklabels(models_names)
ax.set_ylim(0.6, 0.9)
ax.legend()

for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.1%}', ha='center', fontsize=9)
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.1%}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('f1_comparison.png', dpi=150)
plt.close()
print("Saved: f1_comparison.png")

# plot 4: confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, r in enumerate(results):
    cm = confusion_matrix(y_test, r['socio_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    axes[i].set_title(r['Model'])
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.suptitle('Confusion Matrices (Socioeconomic Features)', y=1.02)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrices.png")
