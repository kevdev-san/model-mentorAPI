import pandas as pd
import uuid
from pathlib import Path

DATASET_PATH = Path("app/data/datasets")
PROCESSED_PATH = Path("app/data/processed")


async def save_dataset(file):
    DATASET_PATH.mkdir(parents=True, exist_ok=True)

    dataset_id = str(uuid.uuid4())
    file_path = DATASET_PATH / f"{dataset_id}.csv"

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return dataset_id


def save_processed_dataset(original_id: str, df: pd.DataFrame):
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    new_id = f"{original_id}_processed"
    file_path = PROCESSED_PATH / f"{new_id}.csv"

    df.to_csv(file_path, index=False)

    return new_id


def load_dataset(dataset_id: str) -> pd.DataFrame:
    # Si es un dataset procesado, buscarlo en la carpeta processed/
    if dataset_id.endswith("_processed"):
        file_path = PROCESSED_PATH / f"{dataset_id}.csv"
    else:
        file_path = DATASET_PATH / f"{dataset_id}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' no encontrado")

    return pd.read_csv(file_path)




