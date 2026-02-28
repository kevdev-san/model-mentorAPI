from fastapi import APIRouter, HTTPException
from app.services.dataset_service import load_dataset
from app.services.analysis_service import analyze_dataset

router = APIRouter()

@router.get("/summary/{dataset_id}")
def dataset_analysis(dataset_id: str):
    try:
        df = load_dataset(dataset_id)
        analysis = analyze_dataset(df)

        return {
            "dataset_id": dataset_id,
            "analysis": analysis
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))