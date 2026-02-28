from fastapi import FastAPI
from sklearn import preprocessing
from app.api.routes import dataset, analysis, eda, predict, preproccessing, training



app = FastAPI(
    title="ModelMentor API",
    description="API para análisis automático de datasets y entrenamiento de modelos ML",
    version="0.1.0"
)


#Registrar rutas
app.include_router(dataset.router, prefix="/api/dataset", tags=["Dataset"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(eda.router, prefix="/api/eda", tags=["EDA"])
app.include_router(preproccessing.router, prefix="/api/preprocessing", tags=["Preprocessing"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(predict.router, prefix="/api/predict", tags=["Predict"])

@app.get("/")
def root():
    return {
        "message": "ModelMentor API running 🚀",
        "status": "OK"
    }