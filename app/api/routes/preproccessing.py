from fastapi import APIRouter, HTTPException
from app.services.dataset_service import load_dataset, save_processed_dataset
from app.services.preprocessing_service import preprocess_dataset

router = APIRouter()

@router.post("/run/{dataset_id}")
def run_preprocessing(dataset_id: str):
    try:
        df = load_dataset(dataset_id)

        processed_df, report = preprocess_dataset(df)

        new_id = save_processed_dataset(dataset_id, processed_df)

        return {
            "original_dataset_id": dataset_id,
            "processed_dataset_id": new_id,
            "report": report
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))