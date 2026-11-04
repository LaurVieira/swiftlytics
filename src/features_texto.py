import re
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RAIZ = Path(__file__).resolve().parent.parent
LETRAS = RAIZ / "data" / "raw" / "taylor_swift_lyrics.csv"
MUSICAS = RAIZ / "data" / "processed" / "songs_clean.csv"
SAIDA = RAIZ / "data" / "processed" / "lyrics_features.csv"

def normalizar(titulo: str) -> str:
    t = str(titulo).lower().replace("’", "'")
    t = re.sub(r"\(.*?\)", " ", t)  
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return " ".join(t.split())

def features_texto(texto: str, analisador: SentimentIntensityAnalyzer) -> dict:
    palavras = re.findall(r"[a-z']+", texto.lower())
    linhas = [l.strip().lower() for l in texto.splitlines() if l.strip()]
    n = len(palavras)
    sentimento = analisador.polarity_scores(texto)
    return {
        "n_words": n,
        "lexical_diversity": len(set(palavras)) / n if n else None,
        "repeated_lines": 1 - len(set(linhas)) / len(linhas) if linhas else None,
        "vader_pos": sentimento["pos"],
        "vader_neg": sentimento["neg"],
        "vader_compound": sentimento["compound"],
    }

if __name__ == "__main__":
    letras = pd.read_csv(LETRAS)
    letras.columns = letras.columns.str.strip()
    letras["track_key"] = letras["Title"].map(normalizar)
    letras = letras.drop_duplicates("track_key")

    analisador = SentimentIntensityAnalyzer()
    feats = letras["Lyrics"].apply(lambda t: pd.Series(features_texto(str(t), analisador)))
    feats = pd.concat([letras[["track_key"]], feats], axis=1)

    musicas = pd.read_csv(MUSICAS)
    musicas["track_key"] = musicas["track_name"].map(normalizar)
    juntas = musicas[["track_key", "album_name"]].merge(feats, on="track_key", how="left")
    juntas["com_letra"] = juntas["n_words"].notna()

    print(f"Músicas com letra: {juntas['com_letra'].sum()} de {len(juntas)}")
    print(juntas.groupby("album_name")["com_letra"].agg(["sum", "count"]))

    feats.to_csv(SAIDA, index=False)
    print(f"Features salvas em {SAIDA}")