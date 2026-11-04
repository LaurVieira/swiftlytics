from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data" / "processed" / "songs_clean.csv"
FIGURAS = RAIZ / "reports" / "figures"

ORDEM = ["Taylor Swift", "Fearless (Taylor's Version)", "Speak Now",
         "Red (Taylor's Version)", "1989", "reputation", "Lover",
         "folklore", "evermore", "Midnights"]

FEATURES_CORR = ["valence", "danceability", "energy", "loudness", "speechiness",
                 "acousticness", "liveness", "tempo", "duration_ms"]

def grafico_evolucao(df: pd.DataFrame) -> None:
    medias = (df.groupby("album_name")[["valence", "energy", "acousticness", "danceability"]]
                .mean().loc[ORDEM])
    ax = medias.plot(marker="o", figsize=(11, 5))
    ax.set_xticks(range(len(ORDEM)))
    ax.set_xticklabels(ORDEM, rotation=45, ha="right")
    ax.set_xlabel("")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("Evolução média das features por álbum (ordem das eras)")
    plt.tight_layout()
    plt.savefig(FIGURAS / "evolucao_por_album.png", dpi=150)
    plt.close()

def grafico_correlacao(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[FEATURES_CORR].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlação entre as features de áudio")
    plt.tight_layout()
    plt.savefig(FIGURAS / "correlacao.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    FIGURAS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DADOS)
    grafico_evolucao(df)
    grafico_correlacao(df)
    print(f"Gráficos salvos em {FIGURAS}")