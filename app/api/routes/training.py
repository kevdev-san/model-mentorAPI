from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.dataset_service import load_dataset
from app.services.training_service import train_model


router = APIRouter()


class TrainRequest(BaseModel):
    dataset_id: str
    target: str
    algorithm: str   # linear, logistic, random_forest


@router.post("/train")
def train(req: TrainRequest):
    try:
        df = load_dataset(req.dataset_id)
        result = train_model(df, req.target, req.algorithm)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))