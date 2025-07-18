# 💳 Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning. With over 280,000 anonymized transactions and only 0.17% fraud cases, the dataset presents a significant class imbalance challenge. Our goal is to build accurate and reliable models that can identify fraudulent behavior while minimizing false positives.

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: Time, Amount, 28 anonymized features (V1–V28), and a binary `Class` label (0 = legitimate, 1 = fraud)

## 🚀 Project Workflow

1. **Exploratory Data Analysis (EDA)**  
   - Distribution of transactions  
   - Correlation analysis  
   - Visualization of class imbalance

2. **Preprocessing & Feature Engineering**  
   - Scaling `Amount` and `Time` features  
   - Handling class imbalance using:
     - SMOTE (Synthetic Minority Oversampling)
     - Random undersampling

3. **Model Building**  
   - Trained and compared:
     - Logistic Regression  
     - Random Forest  
     - XGBoost

4. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-Score  
   - ROC-AUC Curve  
   - Confusion Matrix  
   - Precision-Recall Curve

## 🧠 Key Results

| Model              | Accuracy | ROC-AUC | Precision | Recall |
|-------------------|----------|---------|-----------|--------|
| Logistic Regression | 97.7%    | 0.93    | 0.76      | 0.62   |
| Random Forest       | 99.9%    | 0.95    | 0.92      | 0.88   |
| XGBoost             | 99.9%    | 0.95    | 0.93      | 0.91   |

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

## License
[MIT](LICENSE)

## Contact
Made with ❤️ by [Divyam Bansal](https://github.com/divyam-bansal-121)

## 📦 Tech Stack

- **Languages**: Python
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imbalanced-learn`
- **Version Control**: Git & GitHub

