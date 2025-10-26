import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from model import train_and_evaluate, predict_batch, explain_with_shap
from utils import load_data, quick_overview, plot_class_distribution, plot_amount_distribution, plot_correlation_heatmap

# Page config
st.set_page_config(page_title="Fraud Detection Explorer", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("🕵️ Fraud Detection Explorer")
st.markdown("Detect and explore fraudulent transactions using XGBoost + SMOTE. Adjust hyperparameters in the sidebar.")

# --- Sidebar ---
with st.sidebar.expander("Data Options", expanded=True):
    uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])
    use_sample = st.checkbox(
        "Use sample data (data/creditcard.csv)", value=True if uploaded_file is None else False
    )
    st.info("Sample dataset contains anonymized credit card transactions with 'Class' as fraud label.")

with st.sidebar.expander("XGBoost Hyperparameters", expanded=True):
    n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 200, step=50)
    max_depth = st.slider("Maximum Tree Depth", 3, 12, 6)
    learning_rate = st.selectbox("Learning Rate", [0.01, 0.05, 0.1, 0.2], index=2)
    random_state = 42

# Load dataset with caching
@st.cache_data(show_spinner=False)
def get_df(path_or_file):
    if path_or_file is None:
        return None
    return load_data(path_or_file)

df = get_df(
    uploaded_file if uploaded_file is not None else ("data/creditcard.csv" if use_sample else None)
)

if df is None:
    st.warning("No dataset provided. Upload a CSV or place `creditcard.csv` in the `data/` folder.")
    st.stop()

# --- Top KPIs ---
st.markdown("## Dataset Overview")
col1, col2, col3, col4 = st.columns([2,1,1,1])
col1.dataframe(df.head(10))
col2.metric("Total transactions", f"{len(df):,}")
col3.metric("Total frauds", f"{int(df['Class'].sum()):,}", delta_color="inverse")
col4.metric("Fraud rate", f"{df['Class'].mean()*100:.4f}%", delta_color="inverse")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "EDA", "Insights", "Model & Eval", "Predict"])


# --- Overview Tab ---
with tab1:
    st.subheader("Quick Overview")
    st.markdown("High-level statistics and first look at the dataset.")
    quick_overview(df)

    st.subheader("Class Distribution")
    st.pyplot(plot_class_distribution(df))

    st.subheader("Transaction Amount Distribution")
    st.pyplot(plot_amount_distribution(df))

# --- Insights Tab ---
with tab2:
    st.subheader("🔍 Exploratory Data Analysis (EDA)")
    st.markdown("Dive deeper into the dataset — check for missing values, explore distributions, correlations, and outliers.")

    # --- Basic Summary ---
    st.markdown("### Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Fraud Rate", f"{df['Class'].mean()*100:.4f}%")

    st.dataframe(df.describe().T)

    # --- Missing Values ---
    st.markdown("### Missing Values per Column")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("✅ No missing values detected!")
    else:
        st.bar_chart(missing)

    # --- Duplicates ---
    st.markdown("### Duplicate Records")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        st.warning(f"⚠️ Found {duplicates} duplicate rows.")
    else:
        st.success("✅ No duplicate rows found.")

    # --- Outlier Detection ---
    st.markdown("### Outlier Detection (Boxplot of Transaction Amount)")
    fig, ax = plt.subplots()
    sns.boxplot(x=df["Amount"], ax=ax, color="skyblue")
    ax.set_title("Transaction Amount Outliers")
    st.pyplot(fig)

    # --- Feature Distributions ---
    st.markdown("### Feature Distributions")
    numeric_cols = [c for c in df.select_dtypes(include='number').columns if c != "Class"]
    selected_feature = st.selectbox("Select feature to visualize", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], bins=50, kde=True, ax=ax, color="purple")
    ax.set_title(f"Distribution of {selected_feature}")
    st.pyplot(fig)

    # --- Correlation with Target ---
    st.markdown("### Correlation with Fraud Class")
    corr_with_target = df.corr(numeric_only=True)["Class"].sort_values(ascending=False)
    st.dataframe(corr_with_target)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=corr_with_target.values, y=corr_with_target.index, palette="coolwarm", ax=ax)
    ax.set_title("Feature Correlation with Target Class")
    st.pyplot(fig)

    # --- Correlation Heatmap ---
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, cmap="RdBu_r", center=0, ax=ax)
    st.pyplot(fig)

    # --- Summary Insights ---
    st.markdown("### Observations & Insights")
    st.info(
        "💡 **EDA Notes:**\n"
        "- The dataset is highly imbalanced (fraud rate is typically < 1%).\n"
        "- Some features (like V17, V14, V12) often show strong correlation with the fraud label.\n"
        "- Transaction amounts vary widely — check for outliers or normalization needs.\n"
        "- No missing values in the standard dataset, but uploaded data may differ."
    )

# --- Model & Eval Tab ---
with tab3:
    st.subheader("Train XGBoost Model (with SMOTE)")
    st.markdown("Train a model on the dataset. SMOTE is applied to balance the classes.")

    if st.button("Train Model"):
        with st.spinner("Training the XGBoost model..."):
            model_obj = train_and_evaluate(
                df,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state
            )
            joblib.dump(model_obj, "trained_model.joblib")
            st.success("Training finished! Model saved to `trained_model.joblib`.")

            # Metrics & plots
            st.json(model_obj["metrics"])
            st.markdown("**Confusion Matrix**")
            st.pyplot(model_obj["plots"]["confusion_matrix"])
            st.markdown("**ROC Curve**")
            st.pyplot(model_obj["plots"]["roc_curve"])
            st.markdown("**Precision-Recall Curve**")
            st.pyplot(model_obj["plots"]["pr_curve"])

            with open("trained_model.joblib", "rb") as f:
                st.download_button("Download trained_model.joblib", f, file_name="trained_model.joblib")

# --- Predict Tab ---
with tab4:
    st.subheader("Predict New Transactions")
    st.markdown("Upload new transaction data and predict fraud probability using a trained model.")
    model_file = st.file_uploader("Upload a trained model (.joblib) (optional)", type=["joblib"])
    
    try:
        trained = joblib.load(model_file) if model_file else joblib.load("trained_model.joblib")
    except:
        trained = None

    uploaded_new = st.file_uploader("Upload new transactions CSV for prediction", type=["csv"], key="predict")

    if uploaded_new is not None and trained is not None:
        new_df = pd.read_csv(uploaded_new)

        # Prediction
        preds, probs = predict_batch(
            trained["model"],
            trained["scaler"],
            trained["numeric_cols"],
            new_df
        )
        out = new_df.copy()
        out["predicted_class"] = preds
        out["fraud_probability"] = probs

        st.dataframe(out.head(20))
        st.download_button(
            "Download predictions",
            out.to_csv(index=False),
            file_name="predictions.csv"
        )

        # SHAP explanations
        if st.checkbox("Show SHAP explanations for new data"):
            shap_figs = explain_with_shap(
                trained["model"],
                trained["scaler"],
                trained["numeric_cols"],
                new_df,
                plot_type="summary"  # Use summary plot instead of per-row waterfall
            )
            st.markdown("### SHAP Feature Importance")
            for fig in shap_figs:
                st.pyplot(fig)

    elif uploaded_new is not None and trained is None:
        st.warning("No trained model found. Train a model first or upload a `.joblib` model.")
