from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.predict_service import predict

router = APIRouter()


class PredictRequest(BaseModel):
    model_id: str
    input_data: dict


@router.post("/predict")
def run_prediction(req: PredictRequest):
    try:
        result = predict(req.model_id, req.input_data)

        return {
            "model_id": req.model_id,
            "prediction": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))