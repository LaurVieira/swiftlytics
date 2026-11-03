from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data" / "processed" / "songs_clean.csv"

FEATURES = ["danceability", "energy", "loudness", "speechiness",
            "acousticness", "liveness", "tempo", "duration_ms", "mode"]
ALVO = "valence"


if __name__ == "__main__":
    df = pd.read_csv(DADOS)
    X, y = df[FEATURES], df[ALVO]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    modelos = {
        "Média (base)": DummyRegressor(strategy="mean"),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    }

    for nome, modelo in modelos.items():
        r2 = cross_val_score(modelo, X, y, cv=cv, scoring="r2")
        mae = -cross_val_score(modelo, X, y, cv=cv, scoring="neg_mean_absolute_error")
        print(f"{nome:15s} R² = {r2.mean():.2f} (±{r2.std():.2f}) | erro médio = {mae.mean():.3f}")