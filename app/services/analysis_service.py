import pandas as pd


def safe_int(x):
    try:
        return int(x)
    except:
        return None


def safe_float(x):
    try:
        return float(x)
    except:
        return None


def analyze_dataset(df: pd.DataFrame):
    if df.empty:
        return {
            "error": "El dataset está vacío",
            "columns": [],
            "rows": 0
        }

    columns = df.columns.tolist()

    dtypes = {col: str(df[col].dtype) for col in columns}

    nulls_raw = df.isnull().sum()
    nulls = {col: safe_int(nulls_raw[col]) for col in columns}

    total_rows = len(df)

    null_percentages = {}
    for col in columns:
        if total_rows > 0:
            null_percentages[col] = safe_float((nulls[col] / total_rows) * 100)
        else:
            null_percentages[col] = 0.0

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    stats = {}

    for col in numeric_columns:
        series = pd.to_numeric(df[col], errors="coerce")

        stats[col] = {
            "mean": safe_float(series.mean()),
            "std": safe_float(series.std()),
            "min": safe_float(series.min()),
            "max": safe_float(series.max()),
            "median": safe_float(series.median())
        }

    #  sugerencias de posibles variables objetivo (NO decisión automática)
    suggested_targets = []
    for col in columns:
        try:
            unique_vals = safe_int(df[col].nunique())
            if unique_vals and 1 < unique_vals <= 15:
                suggested_targets.append(col)
        except:
            pass

    summary_text = (
        f"El dataset contiene {len(df)} filas y {len(columns)} columnas. "
        f"Se detectaron {sum(nulls.values())} valores nulos en total. "
        f"Columnas numéricas: {len(numeric_columns)}. "
        f"Columnas categóricas: {len(categorical_columns)}."
    )

    return {
        "columns": columns,
        "rows": int(len(df)),
        "dtypes": dtypes,
        "nulls": nulls,
        "null_percentages": null_percentages,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "stats": stats,
        "suggested_targets": suggested_targets,
        "summary": summary_text
    }