from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "data" / "raw" / "taylor_album_songs.csv"
SAIDA = RAIZ / "data" / "processed" / "songs_clean.csv"

def limpar(df: pd.DataFrame) -> pd.DataFrame:

    df = df.dropna(subset=["danceability"]).copy()

    df = df.drop(columns=["lyrics"])

    df["album_release"] = pd.to_datetime(df["album_release"])
    df["track_release"] = pd.to_datetime(df["track_release"])
    df["explicit"] = df["explicit"].astype(bool)
    return df

if __name__ == "__main__":
    bruto = pd.read_csv(ENTRADA)
    limpo = limpar(bruto)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    limpo.to_csv(SAIDA, index=False)
    print(f"Bruto: {bruto.shape} -> Limpo: {limpo.shape}")
    print(f"Salvo em {SAIDA}")