# CS7331-Titanic-Project
The objective of this project is to apply predictive analytics and data mining techniques to the Titanic passenger dataset to identify key factors that influenced passenger survival during the Titanic disaster. By analyzing passenger attributes such as age, gender, passenger class, fare, and family size, we aim to build classification models that can predict whether a passenger survived. Several data mining algorithms, including Naïve Bayes, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Random Forest, will be implemented and compared to evaluate their predictive performance.

In addition to building predictive models, this project will examine how data quality and preprocessing methods impact model performance. The dataset contains missing values in variables such as Age and Cabin, which may influence prediction accuracy. To address this, we will test different data imputation strategies to determine which approach leads to improved model performance.

Furthermore, we will explore data mining techniques to enhance the dataset. For example, passenger names contain titles such as Mr., Mrs., Miss, and Master, which may provide insights into social status or passenger demographics. We will extract these titles and evaluate whether they improve predictive accuracy. Through this analysis, we aim to better understand how demographic, socioeconomic, and data preprocessing factors influence survival outcomes in the Titanic dataset.

## Team & Roles

| Member | Focus Area |
|--------|-----------|
| Jeena Khatri | Socioeconomic feature impact & title extraction |
| Everett Benally | Random Forest & SVM model performance |
| Arturo Perez Espinosa | KNN & Naive Bayes; family size impact on survival |

## Project Status

| Component | Status | Owner |
|-----------|--------|-------|
| Socioeconomic feature impact & title extraction | Complete | Jeena |
| Random Forest & SVM model performance | Complete | Everett |
| KNN & Naive Bayes; family size impact on survival | Complete | Arturo |

## Repository Structure

```
CS7331-Titanic-Project/
├── README.md
├── feature_engineering.py      # Feature engineering, preprocessing, model training & evaluation
├── data/
│   ├── train.csv               # Kaggle Titanic training set (891 rows)
│   └── test.csv                # Kaggle Titanic test set (418 rows)
├── survival_by_title.png       # Generated: survival rate by title group
├── accuracy_comparison.png     # Generated: accuracy comparison across models
├── f1_comparison.png           # Generated: F1 score comparison across models
└── confusion_matrices.png      # Generated: confusion matrices for all models
```

## Run Instructions
1. Put `train.csv` in a local folder named `data` inside the repo: `CS7331-Titanic-Project/data/train.csv`.
2. Install dependencies:
   ```
   python3 -m pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the script:
   ```
   python feature_engineering.py
   ```
4. The script will generate these files:
   - `survival_by_title.png`
   - `accuracy_comparison.png`
   - `f1_comparison.png`
   - `confusion_matrices.png`

---

## Jeena's Work: Feature Engineering & Socioeconomic Impact

### What `feature_engineering.py` does

- Extracts titles (Mr, Mrs, Miss, Master, etc.) from the Name column and groups rare ones into "Rare"
- Fills missing Age with the median age for each title group instead of just the overall median
- Fills missing Embarked with the most common value, drops Cabin since ~80% is missing
- One-hot encodes Title and Embarked (nominal categoricals); label-encodes Sex (binary)
- Scaling is done inside a `Pipeline` per feature set so no data leaks into cross-validation
- Stratified 80/20 train/test split to preserve class balance (38% survivors)
- Compares 3 feature sets:
  - **Base**: just Age + Sex
  - **Socioeconomic**: Age + Sex + Pclass + Fare + Title
  - **All**: every available feature
- Tunes each model (NB, KNN, SVM, RF) with GridSearchCV using 5-fold CV on F1
- Outputs accuracy, F1, precision, recall, confusion matrices, classification reports, and 4 plots

### Results

| Model | Base Acc | Socio Acc | Gain |
|-------|:---:|:---:|:---:|
| Naive Bayes | 77.7% | 78.2% | +0.6% |
| KNN | 76.0% | 82.7% | +6.7% |
| SVM | 77.1% | 81.6% | +4.5% |
| Random Forest | 73.7% | 83.8% | +10.1% |

Random Forest did the best overall — 83.8% accuracy and 0.785 F1 with socioeconomic features. Adding Pclass, Fare, and Title on top of Age/Sex gave a 5–10% boost for most models.

Note: Naive Bayes barely moved (+0.6%) — its feature independence assumption means adding correlated features like Pclass and Fare provides little benefit.
