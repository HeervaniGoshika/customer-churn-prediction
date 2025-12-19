import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Churn_Prediction")

def train():
    df = pd.read_csv("data/processed/churn_processed.csv")
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, "models/model.pkl")

if __name__ == "__main__":
    train()
