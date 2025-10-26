 Fraud Detection Explorer  

An interactive Streamlit dashboard for exploring, analyzing, and detecting fraudulent transactions using XGBoost and SMOTE.

 Overview

Fraudulent transactions are rare but costly. This app provides an **interactive environment** to:
- Perform **Exploratory Data Analysis (EDA)** on transaction data.  
- Visualize class imbalance, feature distributions, and correlations.  
- **Train and evaluate** an XGBoost model with SMOTE to handle imbalanced data.  
- Predict fraud probabilities for new transactions.  
- Interpret predictions using **SHAP (SHapley Additive exPlanations)**.

All wrapped into an intuitive Streamlit dashboard that helps data scientists, analysts, and fraud investigators explore their datasets in depth.

---

Features

- **EDA Dashboard:**  
  Explore missing values, outliers, feature distributions, and correlations.  

- **Model Training:**  
  Train an XGBoost model directly in the app with adjustable hyperparameters.  

- **Performance Metrics:**  
  View confusion matrix, ROC curve, and precision-recall curves interactively.  

- **Prediction Mode:**  
  Upload new transactions and predict fraud probabilities instantly.  

- **Explainability:**  
  Visualize SHAP summary plots to understand which features drive fraud predictions.

---

## 🧩 Project Structure

fraud-detection/
│
├── app.py # Streamlit dashboard
├── model.py # Model training and SHAP logic
├── utils.py # Helper utilities (EDA plots, etc.)
├── data/
│ └── creditcard.csv # Sample dataset (if available)
├── trained_model.joblib # Saved model after training (optional)
├── requirements.txt # Python dependencies
└── README.md # You are here 🚀

## ⚙️ Installation & Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/<your-username>/fraud-detection.git
   cd fraud-detection

2.Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate


3.Install dependencies:

pip install -r requirements.txt


4.Run the app:

streamlit run app.py




📊 Example Screenshots


🧠 Tech Stack


Frontend: Streamlit

Modeling: XGBoost, SMOTE (imbalanced-learn)

EDA & Visualization: Pandas, Matplotlib, Seaborn

Explainability: SHAP


💡 Future Improvements

Add time-based or geolocation fraud pattern analysis

Integrate live data ingestion API

Deploy to Streamlit Cloud or Hugging Face Spaces

Add a model performance comparison tab (XGBoost vs. Logistic Regression)
