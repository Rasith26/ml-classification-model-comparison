import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

st.set_page_config(
    page_title="Online Shopping Purchase Prediction",
    layout="wide"
)

st.title("Online Shoppers Purchasing Intention")
st.write(
    "Compare five machine learning classification models "
    "for predicting online shopping purchase intention."
)

# Upload test data
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if "Revenue" not in data.columns:
        st.error("Uploaded CSV must contain Revenue target column.")
        st.stop()

    X_test = data.drop("Revenue", axis=1)
    y_test = data["Revenue"]

    # Select model
    model_name = st.selectbox(
        "Select Classification Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Gaussian Naive Bayes",
            "SVM"
        ]
    )

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
        "SVM": "svm.joblib"
    }

    model_path = MODEL_DIR / model_files[model_name]

    if not model_path.exists():
        st.error("Model file not found. Train the models first.")
        st.stop()

    model = joblib.load(model_path)

    # Generate predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)

    auc = roc_auc_score(y_test, probabilities)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    mcc = matthews_corrcoef(y_test, predictions)

    st.subheader("Model Evaluation Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1-score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # Confusion matrix
    st.subheader("Confusion Matrix")

    matrix = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots()

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix
    )

    display.plot(ax=ax)

    st.pyplot(fig)

    # Classification report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose()
    )

else:
    st.info("Please upload the labelled test CSV file.")