import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
mlflow.set_tracking_uri('https://dagshub.com/AhmadSabani475/Eksperimen_SML_Ahmad-Rizki-Sabani.mlflow')
mlflow.set_experiment("Eksperimen_kriteria3")

df = pd.read_csv("data_preprocessing.csv")
X = df.drop(columns=['Sleep Disorder'])
y = df['Sleep Disorder']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.sklearn.autolog()

model = RandomForestClassifier(n_estimators=500, max_depth=40, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

joblib.dump(model, 'model.pkl')
print("Model saved to model.pkl")