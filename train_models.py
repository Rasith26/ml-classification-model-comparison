import pandas as pd
import joblib
import json

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Project directories
ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
TEST_DIR = ROOT / "test_data"

MODEL_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save test data
test_data = X_test.copy()
test_data["Revenue"] = y_test
test_data.to_csv(TEST_DIR / "test_data.csv", index=False)

# Identify numerical and categorical columns
numeric_columns = X.select_dtypes(include=["number"]).columns
categorical_columns = X.select_dtypes(exclude=["number"]).columns

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Define classification models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "KNN": KNeighborsClassifier(n_neighbors=7),

    "Gaussian Naive Bayes": GaussianNB(),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
}

# Train and save models
for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    print("\nTraining:", name)
    pipeline.fit(X_train, y_train)
    filename = name.lower().replace(" ", "_") + ".joblib"

    joblib.dump(
        pipeline,
        MODEL_DIR / filename
    )

    print(name, "trained and saved successfully.")

metadata = {
    "target": "Revenue",
    "feature_columns": list(X.columns),
    "random_state": 42,
    "test_size": 0.20
}

with open(MODEL_DIR / "metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

print("\nAll five models trained successfully.")