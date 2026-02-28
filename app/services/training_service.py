import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, root_mean_squared_error
from joblib import dump
from pathlib import Path


MODEL_PATH = Path("app/models/trained")
MODEL_PATH.mkdir(parents=True, exist_ok=True)


def train_model(df: pd.DataFrame, target: str, algorithm: str):
    if target not in df.columns:
        raise ValueError("La variable objetivo no existe en el dataset")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = None
    problem_type = "regression"

    #  detectar tipo de problema
    if y.nunique() <= 10:
        problem_type = "classification"

    #  seleccionar modelo
    if problem_type == "regression":
        if algorithm == "linear":
            model = LinearRegression()
        elif algorithm == "random_forest":
            model = RandomForestRegressor()
        else:
            raise ValueError("Algoritmo no soportado para regresión")

    if problem_type == "classification":
        if algorithm == "logistic":
            model = LogisticRegression(max_iter=1000)
        elif algorithm == "random_forest":
            model = RandomForestClassifier()
        else:
            raise ValueError("Algoritmo no soportado para clasificación")

    #  entrenamiento
    model.fit(X_train, y_train)

    #  evaluación
    y_pred = model.predict(X_test)

    metrics = {}

    if problem_type == "regression":
        metrics["rmse"] = float(root_mean_squared_error(y_test, y_pred))
        metrics["r2"] = float(r2_score(y_test, y_pred))

    if problem_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))

    #  guardar modelo
    model_id = f"{algorithm}_{target}.pkl"
    model_file = MODEL_PATH / model_id
    dump(model, model_file)

    return {
        "model_id": model_id,
        "problem_type": problem_type,
        "algorithm": algorithm,
        "metrics": metrics
    }