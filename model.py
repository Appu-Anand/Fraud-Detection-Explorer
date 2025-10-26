# model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    auc
)
import shap

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_Xy(df, target_col="Class", scaler=None, fit_scaler=True):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        X[numeric_cols] = scaler.transform(X[numeric_cols])
    return X, y, scaler, numeric_cols

# -----------------------------
# Plot helpers
# -----------------------------
def plot_confusion_matrix(cm, labels=[0,1]):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def plot_roc(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1],"--", color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    return fig

def plot_pr_curve(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AP = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    return fig

# -----------------------------
# Training pipeline
# -----------------------------
def train_and_evaluate(df, test_size=0.2, random_state=42, n_estimators=200, max_depth=6, learning_rate=0.1):
    # Split features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Split first (avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numeric columns
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Apply SMOTE
    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)

    # Predictions
    y_probs = model.predict_proba(X_test)[:,1]
    y_pred = (y_probs >= 0.5).astype(int)

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    roc = roc_auc_score(y_test, y_probs)
    metrics = {
        "classification_report": report,
        "roc_auc": roc,
        "test_size": len(y_test),
        "fraud_rate_test": float(y_test.mean())
    }

    # Plots
    cm_fig = plot_confusion_matrix(confusion_matrix(y_test, y_pred))
    roc_fig = plot_roc(y_test, y_probs)
    pr_fig = plot_pr_curve(y_test, y_probs)

    return {
        "model": model,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "X_test": X_test,
        "y_test": y_test,
        "metrics": metrics,
        "plots": {
            "confusion_matrix": cm_fig,
            "roc_curve": roc_fig,
            "pr_curve": pr_fig
        }
    }

# -----------------------------
# Predict new data
# -----------------------------
def predict_batch(model, scaler, numeric_cols, df):
    """
    Predict fraud class and probability on new data using trained model and fitted scaler.
    """
    df_copy = df.copy()

    # Check missing columns
    missing_cols = [c for c in numeric_cols if c not in df_copy.columns]
    if missing_cols:
        raise ValueError(f"Missing required numeric columns: {missing_cols}")

    # Scale numeric columns in correct order
    df_copy[numeric_cols] = scaler.transform(df_copy[numeric_cols])

    preds = model.predict(df_copy[numeric_cols])
    probs = model.predict_proba(df_copy[numeric_cols])[:,1]
    return preds, probs

# -----------------------------
# SHAP explanations
# -----------------------------
import shap
import matplotlib.pyplot as plt

def explain_with_shap(model, scaler, numeric_cols, df_sample):
    """
    Return list of matplotlib figures for SHAP explanations.
    Works with XGBClassifier.
    """
    # Scale sample using the same scaler as training
    df_scaled = df_sample.copy()
    df_scaled[numeric_cols] = scaler.transform(df_scaled[numeric_cols])

    # Create SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df_scaled[numeric_cols])

    figs = []
    # Waterfall for each sample
    for i in range(len(df_scaled)):
        fig = plt.figure()
        shap.plots.waterfall(shap_values[i], show=False)
        figs.append(fig)
    return figs
