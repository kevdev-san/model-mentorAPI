# ModelMentor API

API REST para análisis automático de datasets y entrenamiento de modelos de Machine Learning. Permite cargar un CSV, explorarlo, preprocesarlo y entrenar un modelo en pocos pasos.

---

## Flujo de uso

```
Subir CSV → Análisis → EDA → Preprocesamiento → Entrenamiento → Predicción
```

---

## Endpoints

###  Dataset
**`POST /api/dataset/upload`**

Sube un archivo CSV al servidor y devuelve un `dataset_id` que se usa en los siguientes pasos.

> Solo se aceptan archivos `.csv`.

---

###  Analysis
**`GET /api/analysis/summary/{dataset_id}`**

Devuelve un resumen estadístico del dataset: tipos de columnas, valores nulos, estadísticas descriptivas y sugerencias de variables objetivo.

---

###  EDA
**`GET /api/eda/run/{dataset_id}`**

Genera gráficos exploratorios del dataset: boxplots para detección de outliers, histogramas de distribución y matriz de correlación.

---

###  Preprocessing
**`POST /api/preprocessing/run/{dataset_id}`**

Limpia y prepara el dataset para el entrenamiento:
- Rellena valores nulos (mediana para numéricas, moda para categóricas)
- Codifica variables categóricas automáticamente (`male/female → 1/0`, `yes/no → 1/0`, etc.)
- Devuelve un `processed_dataset_id` y un reporte detallado de las transformaciones

---

###  Training
**`POST /api/training/train`**

Entrena un modelo con el dataset preprocesado. Devuelve el `model_id` y métricas de evaluación.

**Body:**
```json
{
  "dataset_id": "processed_dataset_id",
  "target": "nombre_de_la_columna_objetivo",
  "algorithm": "linear"
}
```

**Algoritmos disponibles:**

| Tipo | Algoritmos |
|------|-----------|
| Regresión | `linear`, `random_forest` |
| Clasificación | `logistic`, `random_forest` |

> El tipo de problema (regresión o clasificación) se detecta automáticamente según la variable objetivo.

---

###  Predict
**`POST /api/predict/predict`**

Usa un modelo entrenado para predecir un valor. Se deben pasar todas las features del dataset **excepto** la variable objetivo, con los valores categóricos ya codificados según el reporte de preprocesamiento.

**Body:**
```json
{
  "model_id": "linear_total_bill.pkl",
  "input_data": {
    "tip": 3.5,
    "sex": 1,
    "smoker": 0,
    "day": 2,
    "time": 0,
    "size": 2
  }
}
```

**Respuesta:**
```json
{
  "model_id": "linear_total_bill.pkl",
  "prediction": [19.64]
}
```

En este ejemplo, el modelo predice que la cuenta total será de **$19.65** a partir de características como la propina, el tamaño de la mesa y si el cliente era fumador.

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/api-modelmentor.git
cd api-modelmentor

# Instalar dependencias
pip install -r requirements.txt

# Iniciar el servidor
uvicorn app.main:app --reload
```

La documentación interactiva estará disponible en `http://127.0.0.1:8000/docs`.

---

## Tecnologías

- **FastAPI** — Framework web
- **scikit-learn** — Modelos de Machine Learning
- **pandas** — Manipulación de datos
- **seaborn / matplotlib** — Visualizaciones
- **joblib** — Serialización de modelos
