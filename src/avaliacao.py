from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data" / "processed" / "songs_clean.csv"

FEATURES = ["danceability", "energy", "loudness", "speechiness",
            "acousticness", "liveness", "tempo", "duration_ms", "mode"]
ALVO = "valence"


if __name__ == "__main__":
    df = pd.read_csv(DADOS)
    X, y, grupos = df[FEATURES], df[ALVO], df["album_name"]

    modelo = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    cv = GroupKFold(n_splits=5)
    r2 = cross_val_score(modelo, X, y, cv=cv, groups=grupos, scoring="r2")
    mae = -cross_val_score(modelo, X, y, cv=cv, groups=grupos,
                           scoring="neg_mean_absolute_error")
    print(f"Ridge (por álbum) R² = {r2.mean():.2f} (±{r2.std():.2f}) | erro médio = {mae.mean():.3f}")

    modelo.fit(X, y)
    coefs = pd.Series(modelo[-1].coef_, index=FEATURES).sort_values()
    print("\nCoeficientes do Ridge:")
    print(coefs.round(3))