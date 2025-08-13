# GOOGLE Stock Price Prediction 📈💰

This repository contains a Python script for predicting the closing price of GOOGLE (GOOGL) stock using machine learning models. The analysis leverages historical stock data and evaluates the performance of two models: Linear Regression and Random Forest Regression.

## Task Objective 🎯

The objective of this project is to predict the next day's closing price of GOOGLE stock based on historical features such as opening price, high, low, and trading volume. The script evaluates the performance of two predictive models using metrics like Mean Squared Error (MSE) and R² Score, and visualizes the actual versus predicted prices.

## Dataset Used 📊

- **Source**: The dataset is sourced from Yahoo Finance using the `yfinance` library.
- **Stock Ticker**: GOOGL (Google)
- **Time Period**: January 1, 2010, to January 1, 2025
- **Features**:
  - `Open`: Opening price of the stock.
  - `High`: Highest price during the trading day.
  - `Low`: Lowest price during the trading day.
  - `Volume`: Number of shares traded.
  - `Close`: Closing price (used to create the target variable).
- **Target Variable**: `Target_Close` (next day's closing price, created by shifting the `Close` column by one day).
- **Data Preprocessing**:
  - Missing values are removed using `dropna()`.
  - The dataset is split into training (80%) and testing (20%) sets without shuffling to preserve the time-series nature.

## Models Applied 🤖

Two machine learning models are implemented to predict the next day's closing price:

1. **Linear Regression**:
   - A simple linear model that assumes a linear relationship between the input features (`Open`, `High`, `Low`, `Volume`) and the target (`Target_Close`).
2. **Random Forest Regressor**:
   - An ensemble model that uses multiple decision trees to capture non-linear relationships and interactions between features.

Both models are trained on the training set and evaluated on the test set.

## Key Results and Findings 📝

The performance of the models is evaluated using **Mean Squared Error (MSE)** and **R² Score**, with the following results:

| Model              | Mean Squared Error | R² Score |
|--------------------|--------------------|----------|
| Linear Regression  | 7.84               | 0.9894   |
| Random Forest      | 173.51             | 0.7661   |

### Detailed Findings 🔍

- **Linear Regression**:
  - **MSE**: 7.84, indicating low prediction error.
  - **R² Score**: 0.9894, suggesting that approximately 98.94% of the variance in the target variable is explained by the model.
  - **Performance**: The model performs exceptionally well, capturing the linear trends in the stock price effectively. The visualization shows that the predicted closing prices closely follow the actual prices, indicating strong predictive power for this dataset.
  
- **Random Forest Regressor**:
  - **MSE**: 173.51, significantly higher than Linear Regression, indicating larger prediction errors.
  - **R² Score**: 0.7661, meaning the model explains about 76.61% of the variance in the target variable.
  - **Performance**: The Random Forest model underperforms compared to Linear Regression. The higher MSE and lower R² suggest that the model struggles to capture the patterns in the data as effectively, possibly due to overfitting or the complexity of the model not being suited for this specific dataset.

### Visualizations 📉

- **Linear Regression Plot**:
  - A line plot compares the actual closing prices with the predicted prices, showing a close alignment between the two.
- **Random Forest Plot**:
  - A similar line plot shows the actual versus predicted prices, but with more noticeable deviations compared to Linear Regression.

### Insights 💡

- The Linear Regression model outperforms the Random Forest model significantly, as evidenced by the lower MSE and higher R² Score. This suggests that the relationship between the input features and the target variable is predominantly linear in this dataset.
- The Random Forest model's poorer performance may indicate that the default hyperparameters are not optimal or that the model is overfitting to noise in the data.
- The visualizations confirm that Linear Regression predictions are more accurate, closely tracking the actual stock prices over time.
- Future improvements could include hyperparameter tuning for the Random Forest model, incorporating additional features (e.g., technical indicators), or exploring other models like LSTM for time-series forecasting.

## Requirements 🛠️

To run the script, you need the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `yfinance`
- `scikit-learn`

Install them using pip:

```bash
pip install pandas numpy matplotlib seaborn yfinance scikit-learn
```

## Files 📂

- `stock_prediction.py`: The main Python script containing the data retrieval, preprocessing, model training, evaluation, and visualization code.
- `README.md`: This file, providing an overview of the project.

## Contributing 🤝

Contributions are welcome! Feel free to open an issue or submit a pull request with improvements, such as additional models, feature engineering, or enhanced visualizations.

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy predicting! 📈🚀
