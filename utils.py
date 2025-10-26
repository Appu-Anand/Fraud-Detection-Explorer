# utils.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(uploaded):
    """
    uploaded: either file-like (from Streamlit uploader) or a path string
    """
    if uploaded is None:
        return None
    try:
        if hasattr(uploaded, "read"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        print("Failed to load data:", e)
        return None
    return df

def quick_overview(df):
    st = __import__("streamlit")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Missing values (per column):")
    st.write(df.isnull().sum().loc[lambda x: x>0])
    st.write("Class balance:")
    st.write(df['Class'].value_counts(normalize=True))

def plot_class_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax)
    ax.set_xticklabels(['Non-Fraud (0)', 'Fraud (1)'])
    ax.set_title("Class distribution")
    return fig

def plot_amount_distribution(df):
    fig, ax = plt.subplots()
    sns.histplot(df['Amount'], bins=50, kde=True, ax=ax)
    ax.set_title("Transaction Amount Distribution")
    return fig

def plot_correlation_heatmap(df):
    # For performance, sample if very large
    sample = df.select_dtypes(include=['number']).sample(n=min(1000, len(df)), random_state=42)
    corr = sample.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Correlation heatmap (sampled)")
    return fig
