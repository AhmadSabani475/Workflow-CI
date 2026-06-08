import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import dagshub

dagshub.init(repo_owner='AhmadSabani475', repo_name='Eksperimen_SML_Ahmad-Rizki-Sabani', mlflow=True)
mlflow.set_experiment("Eksperimen_kriteria3")

df = pd.read_csv("data_preprocessing.csv")
X = df.drop(columns=['Sleep Disorder'])
y = df['Sleep Disorder']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
input_example = X_train[0:5]

mlflow.sklearn.autolog()

with mlflow.start_run():
    n_estimators = 500
    max_depth = 40

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
