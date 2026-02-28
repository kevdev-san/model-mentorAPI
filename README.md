#ENDPOINTS
Dataset
POST
/api/dataset/upload
Upload Dataset
Este endpoint solo recibe archivos csv

Analysis
GET
/api/analysis/summary/{dataset_id}
Dataset Analysis
Este endpoint requiere del id que se crea al subir un dataset

EDA
GET
/api/eda/run/{dataset_id}
Run Eda
Recibe id de Dataset para sacar graficos

Preprocessing
POST
/api/preprocessing/run/{dataset_id}
Run Preprocessing
hace una limpieza de datos, rellena valores nulos

Training
POST
/api/training/train
Train
Se debe pasar el dataset limpio,
el target/nuestra variabla objetivo
y el algoritmo dependiendo si es de regresion opciones: linear, random_forest
o si es de clasificacion: logistic, random_forest

Predict
POST
/api/predict/predict
Run Prediction
Se pasa el modelo entrenado y todas las features menos nuestra variable objetivo
Ejemplo de dataset bill:
{
  "model_id": "linear_totalbill.pkl",
  "input_data": {
    "tip": 3.5,
    "sex": 1,
    "smoker": 0,
    "day": 2,
    "time": 0,
    "size":2
  }
}
Significa que el modelo predice que la cuenta total (total_bill) será de $19.65 
dado los valores que le pasaste como input (propina, tamaño de la mesa, sexo, etc.).
En resumen, le diste las características de una comida y el modelo estimó cuánto 
fue la cuenta total. Si el valor real fuera por ejemplo $21, estaría a menos de $2 
de diferencia, lo cual sería bastante bueno considerando que el RMSE del modelo es de ~$5.57.
