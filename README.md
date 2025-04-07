# 🚢 Titanic Survival Prediction

Welcome to the **Titanic Survival Prediction** repository! This project uses **machine learning and historical passenger data** to predict the survival chances of individuals aboard the RMS Titanic.

## 🚀 Project Overview
This notebook contains a **Logistic Regression model** trained on the famous Titanic dataset to predict survival based on features such as age, sex, ticket fare, and family presence.

The model includes:
- Feature engineering (like extracting titles, combining family size)
- Handling missing values and encoding categorical variables
- Model evaluation using accuracy

## 📊 Data Sources
- [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)
- CSV files: `train.csv` and `test.csv`

## 🧠 How It Works
1. **Data Cleaning**: Missing values in `Age` and `Fare` columns are filled using the median.
2. **Feature Engineering**:
   - Extracted titles from names
   - Created `FamilySize` and `IsAlone` features
3. **Preprocessing**: Converted categorical variables to numerical.
4. **Model Training**: Trained a Logistic Regression model on selected features.
5. **Prediction**: Predicts if a passenger survived (`1`) or not (`0`).
6. **Evaluation**: Assessed using model accuracy (≈79.89%).

### Selected Features:
- `PassengerId`, `Sex`, `Age`, `Fare`, `PortEmbarked`, `Title`, `FamilySize`, `IsAlone`

## ✅ Accuracy
The trained model achieved an accuracy of **79.89%** on the test set.

## 🧾 Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`