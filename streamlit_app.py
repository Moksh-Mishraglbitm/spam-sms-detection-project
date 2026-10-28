
import sys
import subprocess
import importlib.util
from pathlib import Path

# Repair scikit-learn if the deployment environment cannot import it.
try:
    import sklearn.ensemble
except (ImportError, ModuleNotFoundError):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--force-reinstall",
        "scikit-learn==1.6.1"
    ])
    import sklearn.ensemble

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="wide",
)

st.title("📩 Hybrid SMS Spam Detection")
st.write(
    "Logistic Regression + Random Forest + XGBoost"
)

DATA_PATH = Path(__file__).parent / "spam.csv"

@st.cache_resource
def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "spam.csv is missing. Upload it beside streamlit_app.py."
        )

    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "text"]
    elif {"label", "text"}.issubset(df.columns):
        df = df[["label", "text"]].copy()
    else:
        raise ValueError(
            "CSV must contain v1/v2 or label/text columns."
        )

    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)
    df = df[df["label"].isin(["ham", "spam"])]
    df = df.drop_duplicates(subset=["text"])

    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["target"],
        test_size=0.20,
        random_state=42,
        stratify=df["target"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    rf = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
    )

    ensemble = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("rf", rf),
            ("xgb", xgb),
        ],
        voting="soft",
        weights=[4, 3, 3],
    )

    ensemble.fit(X_train_vec, y_train)
    predictions = ensemble.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["HAM", "SPAM"],
        output_dict=True,
        zero_division=0,
    )

    return (
        vectorizer,
        ensemble,
        accuracy,
        matrix,
        report,
        len(df),
    )

try:
    with st.spinner("Training the hybrid models..."):
        vectorizer, model, accuracy, matrix, report, rows = train_model()
except Exception as error:
    st.error(f"Could not start the model: {error}")
    st.stop()

st.success("All three models trained successfully.")

tab1, tab2 = st.tabs(["🔎 Test SMS", "📊 Model Performance"])

with tab1:
    message = st.text_area(
        "Enter an SMS message",
        placeholder="Example: Congratulations! You won a free prize...",
        height=130,
    )

    if st.button("Analyze SMS", type="primary"):
        if not message.strip():
            st.warning("Please enter a message.")
        else:
            features = vectorizer.transform([message])
            probabilities = model.predict_proba(features)[0]
            classes = list(model.classes_)

            spam_probability = float(
                probabilities[classes.index(1)]
            )
            ham_probability = float(
                probabilities[classes.index(0)]
            )

            if spam_probability >= 0.5:
                st.error("🚨 Prediction: SPAM")
            else:
                st.success("✅ Prediction: HAM (likely legitimate)")

            col1, col2 = st.columns(2)
            col1.metric(
                "Spam probability",
                f"{spam_probability * 100:.2f}%",
            )
            col2.metric(
                "Ham probability",
                f"{ham_probability * 100:.2f}%",
            )

            st.progress(spam_probability)
            st.caption(
                "Probabilities are model estimates, not guarantees."
            )

with tab2:
    col1, col2 = st.columns(2)
    col1.metric("Test accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("Dataset messages", f"{rows:,}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["HAM", "SPAM"],
    )
    display.plot(ax=ax, colorbar=False)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")
    st.dataframe(
        pd.DataFrame(report).transpose().round(3),
        use_container_width=True,
    )

st.caption("Project by Moksh Mishra")
