# 🏠 Housing Price Prediction with Linear Regression

## 🎯 Task Objective
The goal of this project is to build a linear regression model to predict housing prices based on features from the Housing Prices Dataset. The model is trained and evaluated using standard regression metrics to gauge its accuracy in predicting house prices. 📈

## 📊 Dataset Used
- **Dataset**: Housing Prices Dataset
- **Source**: Kaggle (`yasserh/housing-prices-dataset`) 🌐
- **Description**: This dataset includes housing features like area, number of bedrooms, bathrooms, and categorical attributes such as furnishing status and parking. The target variable is the house price. Categorical features are encoded using `LabelEncoder` to convert them into numerical values for modeling. 🔢

## 🤖 Models Applied
- **Model**: Linear Regression
  - Implemented using `scikit-learn`'s `LinearRegression` class. 🧠
  - Chosen for its simplicity and interpretability, making it a great baseline for regression tasks.
- **Preprocessing**:
  - Categorical columns (e.g., furnishing status, main road) are encoded with `LabelEncoder` to transform them into numerical values. 🔄
  - Features are split into independent variables (`X`) and the target variable (`y`: price).
- **Training Setup**:
  - Data split: 80% training, 20% testing (`test_size=0.2`, `random_state=42`). ✂️
  - No hyperparameter tuning was performed, as linear regression has minimal hyperparameters.
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE) 📏
  - Root Mean Squared Error (RMSE) 📐
  - R-squared (R2) Score 📊

## 🚀 Key Results and Findings
- **Data Preprocessing**:
  - The dataset was successfully loaded from Kaggle using `kagglehub`. ✅
  - Categorical features were encoded into numerical values, enabling linear regression. 🔧
- **Model Performance**:
  - The linear regression model was trained and evaluated on the test set. 🏋️
  - A scatter plot of actual vs. predicted prices was generated, with a red regression line (y=x) to visualize prediction accuracy. 📉
  - Performance metrics:
    - **Mean Absolute Error (MAE)**: Shows the average absolute difference between predicted and actual prices. 📏
    - **Root Mean Squared Error (RMSE)**: Measures the square root of average squared differences, highlighting larger errors. 📐
    - **R-squared (R2) Score**: Indicates the proportion of variance in house prices explained by the model. 📊
  - Specific values for MAE, RMSE, and R2 are computed and displayed but depend on the dataset and model run.
- **Visualization**:
  - The scatter plot visually assesses how closely predicted prices align with actual prices. 👀
- **Limitations**:
  - Linear regression assumes linear relationships, which may not capture complex patterns in housing prices. ⚠️
  - No feature scaling or advanced feature engineering (e.g., polynomial features) was applied, which could boost performance. 🛠️
  - The model’s simplicity limits its predictive power; more advanced models like Random Forest or XGBoost could improve results. 🌟

This project provides a solid baseline for housing price prediction using linear regression and basic preprocessing on a real-world dataset. Future enhancements could include feature engineering, model selection, or hyperparameter tuning to improve accuracy. 🔮
