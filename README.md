# Diabetes Prediction using Logistic Regression

This project aims to predict whether a patient is likely to have diabetes based on diagnostic measurements. The model is built using the **Logistic Regression algorithm** and the **Pima Indians Diabetes Dataset**. It demonstrates a full machine learning workflow — from data cleaning and model training to evaluation and result interpretation.

---

## Objective

To create a binary classification model that can predict the presence or absence of diabetes using various medical attributes such as glucose level, BMI, insulin level, age, etc.

---

## Dataset

The dataset used is the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) from Kaggle.

**Features:**
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (0 or 1 → non-diabetic or diabetic)

---

## skills Used

| Purpose | Tools |
|--------|-------|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model | Logistic Regression |

---

##  Project Workflow

### 1. Data Loading and Exploration
- Load the CSV dataset using Pandas
- Use `.head()`, `.describe()`, `.info()` to understand the structure
- Check for missing values and anomalies

### 2. Data Preprocessing
- Handle zeros in columns like `Insulin`, `BMI`, `SkinThickness` (if applicable)
- Normalize or scale features if needed (e.g., StandardScaler)
- Separate input features (`X`) and output labels (`y`)

### 3. Train-Test Split
- Split the dataset using `train_test_split()` into 80% training and 20% testing data

### 4. Model Training
- Train a **Logistic Regression** model using `LogisticRegression()` from scikit-learn

### 5. Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Precision, Recall, F1 Score
- Classification Report

### 6. Result Interpretation
- Evaluate which features are most influential
- Analyze false positives and false negatives
- Discuss real-world application scenarios

---

## Sample Results

```text
Accuracy Score: 78%
Confusion Matrix:
[[90 10]
 [19 55]]

Classification Report:
              precision    recall  f1-score   support
           0       0.83      0.90      0.86       100
           1       0.85      0.74      0.79        74
    accuracy                           0.81       174
