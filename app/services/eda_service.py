import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

EDA_PATH = Path("app/data/eda_outputs")


def generate_eda(dataset_id: str, df: pd.DataFrame):
    EDA_PATH.mkdir(parents=True, exist_ok=True)

    dataset_folder = EDA_PATH / dataset_id
    dataset_folder.mkdir(parents=True, exist_ok=True)

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    outputs = {
        "boxplots": [],
        "histograms": [],
        "correlation": None,
        "summary": ""
    }

    #  Boxplots (outliers)
    for col in numeric_columns:
        plt.figure()
        sns.boxplot(x=df[col])
        path = dataset_folder / f"boxplot_{col}.png"
        plt.savefig(path)
        plt.close()
        outputs["boxplots"].append(str(path))

    #  Histogramas
    for col in numeric_columns:
        plt.figure()
        sns.histplot(df[col], kde=True)
        path = dataset_folder / f"hist_{col}.png"
        plt.savefig(path)
        plt.close()
        outputs["histograms"].append(str(path))

    #  Correlación
    if len(numeric_columns) >= 2:
        plt.figure(figsize=(8, 6))
        corr = df[numeric_columns].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        path = dataset_folder / "correlation.png"
        plt.savefig(path)
        plt.close()
        outputs["correlation"] = str(path)

    #  Resumen automático
    outputs["summary"] = (
        f"EDA completado. "
        f"Se analizaron {len(numeric_columns)} columnas numéricas. "
        f"Se generaron boxplots para detección de outliers, histogramas para distribución "
        f"y matriz de correlación para relación entre variables."
    )

    return outputs