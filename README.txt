# Fraud Detection Explorer

Requirements:
- Python 3.8+
- Install dependencies: pip install -r requirements.txt

Project files:
- app.py : Streamlit dashboard
- model.py : model training and evaluation helpers
- utils.py : helper plotting and data loading

How to run:
1. Place your dataset `creditcard.csv` inside the `data/` folder (optional if you will upload via UI).
   You can use the Kaggle Credit Card Fraud Detection dataset or your own transaction dataset; ensure there is a `Class` column (0 = non-fraud, 1 = fraud).
2. In project root:
    pip install -r requirements.txt
    streamlit run app.py
3. Use the sidebar to upload data or use the sample, adjust hyperparameters, train and predict.

Notes:
- The training uses SMOTE on training set only.
- SHAP explainability is included; SHAP may take extra time to compute on large datasets.
- For production use, ensure consistent preprocessing pipeline (scaler) saved alongside model.
