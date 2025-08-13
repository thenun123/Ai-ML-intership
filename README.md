# AI/ML Engineering Internship Tasks – DevelopersHub Corporation (2025) 💼🤖

Welcome to the official repository for my AI/ML Engineering Internship at DevelopersHub Corporation (2025). This repository contains five completed tasks showcasing practical applications of machine learning, data analysis, and natural language processing (NLP) with a focus on real-world datasets and healthcare applications.



## 🚀 Project Overview

Each task in this internship addresses a unique problem in data science and machine learning, ranging from exploratory data analysis to predictive modeling and empathetic chatbot development. The tasks demonstrate proficiency in Python, machine learning libraries, and NLP techniques.

### 📋 Task Summary

| Task No. | Task Name                          | Description                                                                 |
|----------|------------------------------------|-----------------------------------------------------------------------------|
| Task 1   | Iris Dataset Analysis              | Exploratory data analysis and visualization of the Iris dataset using Pandas, Matplotlib, and Seaborn. |
| Task 2   | GOOGLE Stock Price Prediction      | Predicting the next day's closing price of GOOGLE stock using Linear Regression and Random Forest models. |
| Task 3   | Heart Disease Prediction           | Predicting heart disease severity using Decision Tree and Logistic Regression models on a Kaggle dataset. |
| Task 5   | EmpatheticBot: Fine-Tuning a Language Model | Fine-tuning `distilgpt2` with LoRA for empathetic responses using the EmpatheticDialogues dataset. |
| Task 6   | Housing Price Prediction           | Predicting house prices using Linear Regression on the Kaggle Housing Prices Dataset. |

## 🛠️ Tools & Technologies

- **Programming Language**: Python 3.x
- **Environments**: Google Colab, Kaggle Notebooks, VS Code
- **Libraries**:
  - Data Handling: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
  - Data Retrieval: `yfinance`, `kagglehub`
  - NLP: `transformers` (Hugging Face), `datasets`
- **Techniques**:
  - Exploratory Data Analysis (EDA)
  - Regression and Classification Modeling
  - Feature Engineering (Encoding, Scaling, SMOTE)
  - NLP with LoRA Fine-Tuning
- **Hardware**: T4 GPU (for NLP tasks)

## 📁 Repository Structure

```
developershub-aiml-internship-2025/
│
├── Task-1-Iris-Dataset-Analysis/
│   └── iris_analysis.py
├── Task-2-Stock-Price-Prediction/
│   └── stock_prediction.py
├── Task-3-Heart-Disease-Prediction/
│   └── heart_disease_prediction.py
├── Task-5-EmpatheticBot/
│   └── empathetic_bot.py
├── Task-6-Housing-Price-Prediction/
│   └── housing_price_prediction.py
└── README.md
```

## ✅ How to Run the Projects

### Using Google Colab or Kaggle Notebooks (Recommended)
1. Open the respective `.py` or `.ipynb` file in Google Colab or Kaggle.
2. Install required dependencies (listed in each task's README).
3. Run the cells sequentially to execute the analysis, modeling, or chatbot inference.
4. For Task 5, ensure a T4 GPU is enabled for faster training.

### Using VS Code
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/developershub-aiml-internship-2025.git
   ```
2. Navigate to the project directory:
   ```bash
   cd developershub-aiml-internship-2025
   ```
3. Set up a Python environment and install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn yfinance scikit-learn transformers datasets kagglehub torch
   ```
4. Run the desired script:
   ```bash
   python Task-X-.../script_name.py
   ```

## 📌 Important Notes

- **Open-Source Models**: No proprietary APIs (e.g., OpenAI) were used; all models are open-source (e.g., `distilgpt2`, Hugging Face Transformers).
- **Resource Constraints**: Task 5 (EmpatheticBot) is optimized for Kaggle’s 9 GB RAM and T4 GPU, using LoRA for efficient fine-tuning.
- **Data Sources**: Datasets are sourced from Kaggle (`heart-disease-data`, `housing-prices-dataset`) and Hugging Face (`empathetic_dialogues`), with stock data from Yahoo Finance (`yfinance`).
- **Visualizations**: Each task includes visualizations (e.g., pairplots, scatter plots, ROC curves, confusion matrices) to provide insights into data and model performance.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



## 🤝 Acknowledgements

- **Hugging Face**: For providing access to `distilgpt2` and the EmpatheticDialogues dataset.
- **Kaggle**: For hosting datasets and providing computational resources.
- **DevelopersHub Mentors**: For guidance and support throughout the internship.
- **Open-Source Community**: For tools like `pandas`, `scikit-learn`, and `yfinance`.

Happy exploring and predicting! 🌟📈
