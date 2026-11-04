from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data" / "processed" / "songs_clean.csv"
FIGURAS = RAIZ / "reports" / "figures"

AUDIO = ["danceability", "energy", "loudness", "speechiness",
         "acousticness", "liveness", "tempo", "duration_ms", "mode"]
ALVO = "valence"

ORDEM = ["Taylor Swift", "Fearless (Taylor's Version)", "Speak Now",
         "Red (Taylor's Version)", "1989", "reputation", "Lover",
         "folklore", "evermore", "Midnights"]


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    FIGURAS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DADOS)
    X, y, grupos = df[AUDIO], df[ALVO], df["album_name"]

    linhas = []
    for treino, teste in LeaveOneGroupOut().split(X, y, grupos):
        modelo = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        modelo.fit(X.iloc[treino], y.iloc[treino])
        pred = modelo.predict(X.iloc[teste])
        media_treino = y.iloc[treino].mean()
        real = y.iloc[teste]
        linhas.append({
            "album": grupos.iloc[teste].iloc[0],
            "n": len(teste),
            "valencia_real": real.mean(),
            "valencia_prevista": pred.mean(),
            "mae_modelo": mean_absolute_error(real, pred),
            "mae_base": mean_absolute_error(real, [media_treino] * len(teste)),
        })

    res = pd.DataFrame(linhas).set_index("album").loc[ORDEM]
    res["ganho_mae"] = res["mae_base"] - res["mae_modelo"]
    print(res.round(3).to_string())
    print(f"\nErro médio geral: modelo = {res['mae_modelo'].mean():.3f} | "
          f"base = {res['mae_base'].mean():.3f}")

    ax = res[["valencia_real", "valencia_prevista"]].plot(kind="bar", figsize=(11, 5))
    ax.set_xticklabels(ORDEM, rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Valência média")
    ax.legend(["Real", "Prevista (álbum fora do treino)"])
    plt.title("Valência média por álbum: real vs prevista")
    plt.tight_layout()
    plt.savefig(FIGURAS / "real_vs_previsto_por_album.png", dpi=150)
    plt.close()
    print(f"\nGráfico salvo em {FIGURAS}")