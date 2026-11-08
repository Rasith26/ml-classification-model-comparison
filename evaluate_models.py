import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "model"
TEST_PATH = ROOT / "test_data" / "test_data.csv"

# Load test dataset
test_data = pd.read_csv(TEST_PATH)

X_test = test_data.drop("Revenue", axis=1)
y_test = test_data["Revenue"]

# Model files
models = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "SVM": "svm.joblib"
}

results = []

for name, filename in models.items():

    model_path = MODEL_DIR / filename

    model = joblib.load(model_path)

    # Predictions
    predictions = model.predict(X_test)

    # Probability scores for AUC
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
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

    results.append([
        name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

    print("\nModel:", name)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

# Create comparison table
columns = [
    "Model",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1-score",
    "MCC"
]

results_df = pd.DataFrame(results, columns=columns)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(
    ROOT / "results.csv",
    index=False
)

print("\nEvaluation completed successfully.")