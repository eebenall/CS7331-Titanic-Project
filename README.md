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
- Encodes categorical features and scales everything with StandardScaler
- Compares 3 feature sets:
  - **Base**: just Age + Sex
  - **Socioeconomic**: Age + Sex + Pclass + Fare + Title
  - **All**: every available feature
- Tunes each model (NB, KNN, SVM, RF) with GridSearchCV using 5-fold CV on F1
- Outputs accuracy, F1, confusion matrices, classification reports, and 4 plots

### Results

| Model | Base Acc | Socio Acc | Gain |
|-------|:---:|:---:|:---:|
| Naive Bayes | 78.2% | 77.1% | −1.1% |
| KNN | 76.5% | 82.1% | +5.6% |
| SVM | 79.3% | 84.4% | +5.0% |
| Random Forest | 77.7% | 87.2% | +9.5% |

Random Forest did the best overall — 87.2% accuracy and 0.83 F1 with socioeconomic features. Adding Pclass, Fare, and Title on top of Age/Sex gave a 5–10% boost for most models.

Note: Naive Bayes saw a slight drop with the socioeconomic set, likely because the high correlation between Pclass and Fare violates its feature independence assumption.
