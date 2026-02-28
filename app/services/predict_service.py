import pandas as pd
from joblib import load
from pathlib import Path

MODEL_PATH = Path("app/models/trained")


def predict(model_id: str, input_data: dict):
    model_file = MODEL_PATH / model_id

    if not model_file.exists():
        raise FileNotFoundError("Modelo no encontrado")

    model = load(model_file)

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)

    return prediction.tolist()