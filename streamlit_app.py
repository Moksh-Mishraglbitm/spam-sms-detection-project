import re
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
st.write("Scikit-learn version:", sklearn.__version__)
st.write("Scikit-learn location:", sklearn.__file__)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="wide")
st.title("📩 Hybrid SMS Spam Detection")
st.caption("TF-IDF + Logistic Regression + Random Forest + XGBoost")
st.write("Enter an SMS message to estimate whether it looks like spam. Test-set metrics are also available.")

DATA_PATH = Path(__file__).parent / "spam.csv"

@st.cache_resource(show_spinner="Training the hybrid model on the SMS dataset...")
def train_system(csv_path: str, file_mtime: float):
    data = pd.read_csv(csv_path, encoding="latin-1")
    if {"v1", "v2"}.issubset(data.columns):
        data = data[["v1", "v2"]].copy()
        data.columns = ["label", "text"]
    elif {"label", "text"}.issubset(data.columns):
        data = data[["label", "text"]].copy()
    else:
        raise ValueError("Expected dataset columns v1/v2 or label/text.")

    data = data.dropna(subset=["label", "text"]).copy()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["text"] = data["text"].astype(str)
    data = data[data["label"].isin(["ham", "spam"])]
    data["label"] = data["label"].map({"ham": 0, "spam": 1})
    data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if data["label"].nunique() != 2 or len(data) < 10:
        raise ValueError("The dataset needs both ham and spam examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.20, random_state=42, stratify=data["label"]
    )
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", max_features=10000,
        ngram_range=(1, 2), sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, eval_metric="logloss", random_state=42, n_jobs=2
    )
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
        voting="soft", weights=[4, 3, 3], n_jobs=1
    )
    ensemble.fit(X_train_vec, y_train)
    preds = ensemble.predict(X_test_vec)
    accuracy = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    report = classification_report(
        y_test, preds, labels=[0, 1],
        target_names=["Ham (legitimate)", "Spam"], output_dict=True, zero_division=0
    )
    individual = {}
    for name, estimator in [("Logistic Regression", lr), ("Random Forest", rf), ("XGBoost", xgb)]:
        estimator.fit(X_train_vec, y_train)
        individual[name] = accuracy_score(y_test, estimator.predict(X_test_vec))
    return {
        "vectorizer": vectorizer, "model": ensemble, "accuracy": accuracy,
        "confusion_matrix": cm, "report": report, "individual": individual,
        "rows": len(data), "ham_count": int((data["label"] == 0).sum()),
        "spam_count": int((data["label"] == 1).sum()), "features": X_train_vec.shape[1]
    }

def strong_spam_pattern(message: str):
    # Transparent warning only; this does not change the model's prediction/probability.
    patterns = [
        r"\bfree\s+free\b", r"\bwon\b.*\b(prize|cash|reward|voucher)\b",
        r"\b(claim|collect)\b.*\b(prize|reward|cash|offer)\b",
        r"\b(congratulations|winner|winning)\b", r"\b(click here|call now)\b",
        r"\b\d+%\s*off\b", r"\b(urgent|limited time offer)\b"
    ]
    return any(re.search(p, message.lower()) for p in patterns)

if not DATA_PATH.exists():
    st.error("Dataset `spam.csv` not found. Add it beside `app.py` in your GitHub repository.")
    st.stop()

try:
    system = train_system(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load/train the model: {exc}")
    st.stop()

with st.sidebar:
    st.header("📊 Dataset & model")
    st.metric("Messages", f"{system['rows']:,}")
    st.metric("Ham messages", f"{system['ham_count']:,}")
    st.metric("Spam messages", f"{system['spam_count']:,}")
    st.metric("TF-IDF features", f"{system['features']:,}")
    st.caption("The model trains on startup and is cached for subsequent visits.")

tab_predict, tab_performance, tab_about = st.tabs(["🔎 Test a message", "📈 Model performance", "ℹ️ About"])

with tab_predict:
    st.subheader("Check an SMS")
    examples = [
        "Congratulations! You have won a free prize. Call now to claim your reward.",
        "Hey, are we meeting at the college library today?",
        "free free free!!"
    ]
    chosen = st.selectbox("Try an example", ["Choose an example..."] + examples)
    default_text = "" if chosen == "Choose an example..." else chosen
    message = st.text_area("SMS message", value=default_text, placeholder="Paste or type a message here...", height=130)

    if st.button("Analyze message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Enter a message first.")
        else:
            vec = system["vectorizer"].transform([message])
            probs = system["model"].predict_proba(vec)[0]
            classes = list(system["model"].classes_)
            spam_prob = float(probs[classes.index(1)])
            ham_prob = float(probs[classes.index(0)])
            rule_flag = strong_spam_pattern(message)

            if rule_flag:
                st.warning("⚠️ A suspicious phrase pattern was detected. Treat this message carefully, even if the model predicts HAM.")
            if spam_prob >= 0.5:
                st.error("🚨 Model prediction: SPAM")
            else:
                st.success("✅ Model prediction: HAM (likely legitimate)")

            c1, c2 = st.columns(2)
            c1.metric("Spam probability (model)", f"{spam_prob * 100:.2f}%")
            c2.metric("Ham probability (model)", f"{ham_prob * 100:.2f}%")
            st.progress(min(max(spam_prob, 0.0), 1.0))
            st.caption("Probabilities are model estimates, not guarantees. The phrase-pattern warning is separate and does not change the model prediction.")

with tab_performance:
    st.subheader("Held-out test-set results")
    st.metric("Hybrid ensemble accuracy", f"{system['accuracy'] * 100:.2f}%")
    st.caption("Measured on a stratified 20% test split (random_state=42). Accuracy is not guaranteed.")
    perf_df = pd.DataFrame({
        "Model": list(system["individual"].keys()) + ["Hybrid Ensemble"],
        "Accuracy (%)": [v * 100 for v in system["individual"].values()] + [system["accuracy"] * 100]
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    st.bar_chart(perf_df.set_index("Model"))
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=system["confusion_matrix"], display_labels=["Ham", "Spam"])
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Hybrid Ensemble Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(system["report"]).transpose().round(3), use_container_width=True)

with tab_about:
    st.subheader("How it works")
    st.markdown("""
1. **Dataset:** SMS messages labelled `ham` or `spam`.
2. **Preprocessing:** removes missing labels/messages and duplicate message text.
3. **TF-IDF:** converts text into numeric features using unigrams and bigrams.
4. **Hybrid classifier:** soft-voting ensemble of Logistic Regression, Random Forest, and XGBoost with weights 4:3:3.
5. **Evaluation:** accuracy, classification report, and confusion matrix on a held-out test set.
""")
    st.info("Educational project only. A prediction is not a guarantee that a message is safe.")
    st.write("Project: Hybrid SMS Spam Detection")
    st.write("Author: Moksh Mishra")
