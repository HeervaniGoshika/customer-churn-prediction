import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess():
    df = pd.read_csv("data/raw/churn.csv")

    # Drop identifier columns (never useful for ML)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Separate target
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # Identify categorical & numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    # One-hot encode categorical columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Scale numerical columns only
    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

    # Combine features and target
    processed = X_encoded.copy()
    processed["Churn"] = y.values

    # Save processed data
    processed.to_csv("data/processed/churn_processed.csv", index=False)

    print("✅ Preprocessing completed successfully")

if __name__ == "__main__":
    preprocess()
