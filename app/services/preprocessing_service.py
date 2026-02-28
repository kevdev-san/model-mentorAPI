import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Mapeos explícitos para valores comunes binarios.
# Puedes agregar más según tu dominio.
EXPLICIT_MAPPINGS = {
    frozenset(["male", "female"])           : {"male": 1, "female": 0},
    frozenset(["yes", "no"])                : {"yes": 1, "no": 0},
    frozenset(["true", "false"])            : {"true": 1, "false": 0},
    frozenset(["1", "0"])                   : {"1": 1, "0": 0},
    frozenset(["si", "no"])                 : {"si": 1, "no": 0},
    frozenset(["positive", "negative"])     : {"positive": 1, "negative": 0},
}


def _find_explicit_mapping(series: pd.Series):
    """Devuelve un mapeo explícito si los valores únicos coinciden, si no None."""
    unique_vals = set(series.dropna().str.lower().unique())
    for key_set, mapping in EXPLICIT_MAPPINGS.items():
        if unique_vals == key_set or unique_vals.issubset(key_set):
            return mapping
    return None


def preprocess_dataset(df: pd.DataFrame, strategy="auto"):
    df = df.copy()

    report = {
        "nulls_handled": {},
        "encodings": [],
        "dropped_columns": [],
        "summary": ""
    }

    # Separar tipos
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # ── NULOS numéricos: imputar con mediana ──────────────────────────────────
    for col in numeric_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            report["nulls_handled"][col] = f"Imputado con mediana ({median})"

    # ── NULOS categóricos: imputar con moda o eliminar columna ───────────────
    cols_to_drop = []
    for col in categorical_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode[0])
                report["nulls_handled"][col] = f"Imputado con moda ({mode[0]})"
            else:
                cols_to_drop.append(col)
                report["dropped_columns"].append(col)

    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        categorical_cols = [c for c in categorical_cols if c not in cols_to_drop]

    # ── ENCODING categórico ───────────────────────────────────────────────────
    cols_to_drop_enc = []
    for col in categorical_cols:
        if col not in df.columns:
            continue

        series_lower = df[col].astype(str).str.lower()
        explicit = _find_explicit_mapping(series_lower)

        if explicit:
            # Mapeo explícito: valores conocidos (male/female, yes/no, etc.)
            df[col] = series_lower.map(explicit)
            report["encodings"].append({
                "column": col,
                "method": "explicit_mapping",
                "mapping": explicit
            })

        elif df[col].nunique() <= 10:
            # Pocas categorías: Label Encoding con reporte del mapeo
            le = LabelEncoder()
            try:
                df[col] = le.fit_transform(df[col].astype(str))
                mapping = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
                report["encodings"].append({
                    "column": col,
                    "method": "label_encoding",
                    "mapping": mapping
                })
            except Exception:
                cols_to_drop_enc.append(col)
                report["dropped_columns"].append(col)

        else:
            # Muchas categorías: One-Hot Encoding
            try:
                dummies = pd.get_dummies(df[col], prefix=col).astype(int)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                report["encodings"].append({
                    "column": col,
                    "method": "one_hot_encoding",
                    "new_columns": dummies.columns.tolist()
                })
            except Exception:
                cols_to_drop_enc.append(col)
                report["dropped_columns"].append(col)

    if cols_to_drop_enc:
        existing = [c for c in cols_to_drop_enc if c in df.columns]
        if existing:
            df.drop(columns=existing, inplace=True)

    report["summary"] = (
        f"Preprocesamiento completado. "
        f"{len(report['nulls_handled'])} columnas con nulos imputados. "
        f"{len(report['encodings'])} columnas codificadas. "
        f"{len(report['dropped_columns'])} columnas eliminadas."
    )

    return df, report
