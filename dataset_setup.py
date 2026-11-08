import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Set project directories
ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "online_shoppers_intention.csv"
TEST_PATH = ROOT / "test_data" / "test_data.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Display dataset information
print("Dataset Shape:", df.shape)

print("\nFirst Five Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nTarget Variable Distribution:")
print(df["Revenue"].value_counts())

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

test_data = X_test.copy()
test_data["Revenue"] = y_test

TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
test_data.to_csv(TEST_PATH, index=False)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

print("\nTest dataset saved successfully.")