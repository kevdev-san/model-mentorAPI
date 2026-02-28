from fastapi import APIRouter, HTTPException
from app.services.dataset_service import load_dataset
from app.services.eda_service import generate_eda

router = APIRouter()

@router.get("/run/{dataset_id}")
def run_eda(dataset_id: str):
    try:
        df = load_dataset(dataset_id)
        eda_result = generate_eda(dataset_id, df)

        return {
            "dataset_id": dataset_id,
            "eda": eda_result
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))