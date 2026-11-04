from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features_texto import normalizar

RAIZ = Path(__file__).resolve().parent.parent
MUSICAS = RAIZ / "data" / "processed" / "songs_clean.csv"
TEXTO = RAIZ / "data" / "processed" / "lyrics_features.csv"

AUDIO = ["danceability", "energy", "loudness", "speechiness",
         "acousticness", "liveness", "tempo", "duration_ms", "mode"]
TEXT = ["n_words", "lexical_diversity", "repeated_lines",
        "vader_pos", "vader_neg", "vader_compound"]
ALVO = "valence"


def avaliar(nome, modelo, X, y, grupos):
    cv = GroupKFold(n_splits=5)
    r2 = cross_val_score(modelo, X, y, cv=cv, groups=grupos, scoring="r2")
    mae = -cross_val_score(modelo, X, y, cv=cv, groups=grupos,
                           scoring="neg_mean_absolute_error")
    print(f"{nome:18s} R² = {r2.mean():.2f} (±{r2.std():.2f}) | erro médio = {mae.mean():.3f}")


if __name__ == "__main__":
    musicas = pd.read_csv(MUSICAS)
    musicas["track_key"] = musicas["track_name"].map(normalizar)
    df = musicas.merge(pd.read_csv(TEXTO), on="track_key", how="inner").dropna(subset=TEXT)
    print(f"Músicas usadas: {len(df)} | álbuns: {df['album_name'].nunique()}\n")

    y, grupos = df[ALVO], df["album_name"]

    print("Correlação das features de texto com a valência:")
    print(df[TEXT + [ALVO]].corr()[ALVO].drop(ALVO).round(2), "\n")

    def ridge():
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    avaliar("Média (base)", DummyRegressor(strategy="mean"), df[AUDIO], y, grupos)
    avaliar("Só áudio", ridge(), df[AUDIO], y, grupos)
    avaliar("Só texto", ridge(), df[TEXT], y, grupos)
    avaliar("Áudio + texto", ridge(), df[AUDIO + TEXT], y, grupos)