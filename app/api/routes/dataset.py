from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.dataset_service import save_dataset

router = APIRouter()

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos CSV")

    try:
        dataset_id = await save_dataset(file)

        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "status": "Dataset cargado correctamente"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))