#  Churn Risk Forecasting

This project uses machine learning to predict **customer churn** based on service usage and demographic features, using the Telco Customer Churn dataset from Kaggle.

---

## Dataset

- **Source**: [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Target Column**: `Churn` (Yes/No)
- **Features**: Demographics, subscription type, service usage, and billing details

---

##  Project Workflow

### 1. **Data Cleaning**
- Removed `customerID` column
- Converted `TotalCharges` to float and handled blank values
- Checked and handled missing data

### 2. **Exploratory Data Analysis (EDA)**
- Histograms and box plots for numerical features
- Count plots for categorical features
- Correlation heatmap

### 3. **Preprocessing**
- Applied `LabelEncoder` to all categorical features
- Used `SMOTE` to handle class imbalance in the target column

### 4. **Model Building**
Trained and evaluated the following models using **5-fold cross-validation**:
- `DecisionTreeClassifier`
- `RandomForestClassifier`  *Best performing*
- `XGBRFClassifier` (XGBoost)

### 5. **Evaluation Metrics**
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

### 6. **Model Deployment**
- Saved trained model and label encoders using `pickle`
- Created a test prediction example using saved files

---

##  Best Performing Model

**Random Forest Classifier**
- Gave the highest average cross-validation accuracy
- Generalized well on the test set

---

##  How to Run

1. Clone this repository
2. Install the dependencies:
   ```bash
   pip install pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
3. Run the script
   ```python script.py```
