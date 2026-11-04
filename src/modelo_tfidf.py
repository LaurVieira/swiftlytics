from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features_texto import normalizar

RAIZ = Path(__file__).resolve().parent.parent
MUSICAS = RAIZ / "data" / "processed" / "songs_clean.csv"
LETRAS = RAIZ / "data" / "raw" / "taylor_swift_lyrics.csv"

AUDIO = ["danceability", "energy", "loudness", "speechiness",
         "acousticness", "liveness", "tempo", "duration_ms", "mode"]
ALVO = "valence"
ALPHAS = [1, 10, 100]


def montar(usar_audio: bool, usar_texto: bool, alpha: float):
    partes = []
    if usar_audio:
        partes.append(("audio", StandardScaler(), AUDIO))
    if usar_texto:
        tfidf = TfidfVectorizer(stop_words="english", min_df=3,
                                max_features=300, sublinear_tf=True)
        partes.append(("texto", tfidf, "Lyrics"))
    return make_pipeline(ColumnTransformer(partes), Ridge(alpha=alpha))


def avaliar(nome, modelo, X, y, grupos):
    cv = GroupKFold(n_splits=5)
    r2 = cross_val_score(modelo, X, y, cv=cv, groups=grupos, scoring="r2")
    mae = -cross_val_score(modelo, X, y, cv=cv, groups=grupos,
                           scoring="neg_mean_absolute_error")
    print(f"{nome:28s} R² = {r2.mean():.2f} (±{r2.std():.2f}) | erro médio = {mae.mean():.3f}")


if __name__ == "__main__":
    letras = pd.read_csv(LETRAS)
    letras.columns = letras.columns.str.strip()
    letras["track_key"] = letras["Title"].map(normalizar)
    letras = letras.drop_duplicates("track_key")[["track_key", "Lyrics"]]

    musicas = pd.read_csv(MUSICAS)
    musicas["track_key"] = musicas["track_name"].map(normalizar)
    df = musicas.merge(letras, on="track_key", how="inner").dropna(subset=["Lyrics"])
    print(f"Músicas usadas: {len(df)} | álbuns: {df['album_name'].nunique()}\n")

    X, y, grupos = df[AUDIO + ["Lyrics"]], df[ALVO], df["album_name"]

    avaliar("Média (base)", DummyRegressor(strategy="mean"), df[AUDIO], y, grupos)
    for alpha in ALPHAS:
        avaliar(f"Só áudio (alpha={alpha})", montar(True, False, alpha), X, y, grupos)
        avaliar(f"Só TF-IDF (alpha={alpha})", montar(False, True, alpha), X, y, grupos)
        avaliar(f"Áudio + TF-IDF (alpha={alpha})", montar(True, True, alpha), X, y, grupos)
        print()

    modelo = montar(False, True, 10).fit(X, y)
    palavras = modelo[0].named_transformers_["texto"].get_feature_names_out()
    pesos = pd.Series(modelo[-1].coef_, index=palavras).sort_values()
    print("Palavras ligadas a valência MENOR:", ", ".join(pesos.index[:8]))
    print("Palavras ligadas a valência MAIOR:", ", ".join(pesos.index[-8:]))