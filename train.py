import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("assignment5")

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test))

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("dataset", "synthetic make_classification")
    accuracy = 0.70 # Simulate a model that fails the threshold
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, artifact_path="model")

    with open("model_info.txt", "w") as f:
        f.write(f"{run.info.run_id}\n")
        f.write(f"{accuracy:.4f}\n")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"RUN_ID:{run.info.run_id}")   