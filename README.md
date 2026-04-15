# CS7331-Titanic-Project
The objective of this project is to apply predictive analytics and data mining techniques to the Titanic passenger dataset to identify key factors that influenced passenger survival during the Titanic disaster. By analyzing passenger attributes such as age, gender, passenger class, fare, and family size, we aim to build classification models that can predict whether a passenger survived. Several data mining algorithms, including Naïve Bayes, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Random Forest, will be implemented and compared to evaluate their predictive performance.

In addition to building predictive models, this project will examine how data quality and preprocessing methods impact model performance. The dataset contains missing values in variables such as Age and Cabin, which may influence prediction accuracy. To address this, we will test different data imputation strategies to determine which approach leads to improved model performance.

Furthermore, we will explore data mining techniques to enhance the dataset. For example, passenger names contain titles such as Mr., Mrs., Miss, and Master, which may provide insights into social status or passenger demographics. We will extract these titles and evaluate whether they improve predictive accuracy. Through this analysis, we aim to better understand how demographic, socioeconomic, and data preprocessing factors influence survival outcomes in the Titanic dataset.

## Run instructions
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
