# Heart Disease Prediction Project ❤️🩺

## Task Objective 🎯
The objective of this project is to predict the presence and severity of heart disease in patients using machine learning models. The task involves preprocessing a heart disease dataset, handling missing values, encoding categorical features, scaling numerical features, balancing the dataset, and applying classification models to predict the target variable (`num`), which represents the severity of heart disease (0 to 4).

## Dataset Used 📊💾
The dataset used in this project is the Heart Disease Dataset sourced from Kaggle (redwankarimsony/heart-disease-data). It contains various features related to patient health, such as age, sex, chest pain type, blood pressure, cholesterol levels, and more. The target variable `num` indicates the severity of heart disease, with values ranging from 0 (no heart disease) to 4 (severe heart disease).

## Models Applied 🤖
Two classification models were applied to predict the severity of heart disease:

- **Decision Tree Classifier 🌳**: A tree-based model that splits the data based on feature values to make predictions.
- **Logistic Regression 📈**: A linear model used for multi-class classification, optimized for the dataset with a maximum of 1000 iterations.

## Preprocessing Steps 🛠️
- **Handling Missing Values 🕵️‍♂️**: Numerical columns were filled with their mean values, and categorical columns were filled with their mode.
- **Feature Dropping 🗑️**: Irrelevant or incomplete features (`id`, `ca`, `slope`, `thal`) were dropped.
- **Encoding 🔢**: Categorical features were encoded using `LabelEncoder`.
- **Scaling 📏**: Numerical features were scaled to the range [0, 1] using `MinMaxScaler`.
- **Class Balancing ⚖️**: The dataset was balanced using SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance in the target variable.
- **Train-Test Split ✂️**: The dataset was split into 80% training and 20% testing sets.

## Key Results and Findings 📝🔍

### Decision Tree Classifier 🌳
- **Accuracy ✅**: The model achieved a certain accuracy on the test set (refer to the output for exact value).
- **Classification Report 📋**: Provides precision, recall, and F1-score for each class (0 to 4).
- **ROC Curves 📉**: Multi-class ROC curves were plotted, showing the area under the curve (AUC) for each class, indicating the model's ability to distinguish between classes.
- **Confusion Matrix 🧩**: Visualized to show the distribution of correct and incorrect predictions across classes.

### Logistic Regression 📈
- **Accuracy ✅**: The model achieved a certain accuracy on the test set (refer to the output for exact value).
- **Classification Report 📋**: Provides precision, recall, and F1-score for each class, generally showing balanced performance across classes.
- **ROC Curves 📉**: Multi-class ROC curves were plotted, with AUC values indicating the model's discriminative power.
- **Confusion Matrix 🧩**: Visualized to show prediction performance, typically with fewer misclassifications compared to the Decision Tree.

### Key Findings 🧠
- Both models performed well after preprocessing and balancing the dataset, with Logistic Regression generally showing more consistent performance across classes due to its linear nature.
- SMOTE effectively addressed class imbalance, improving model performance on minority classes.
- The ROC curves and confusion matrices indicate that both models are capable of distinguishing between different levels of heart disease severity, though performance varies by class.
- The dataset's features, after preprocessing, provided sufficient information for accurate predictions, with key features like age, cholesterol, and chest pain type likely contributing significantly to the models' performance.

This project demonstrates the effectiveness of preprocessing techniques and machine learning models in predicting heart disease severity, with potential applications in medical diagnostics. 🚀
