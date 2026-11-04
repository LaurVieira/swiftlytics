from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data" / "processed" / "songs_clean.csv"

AUDIO = ["danceability", "energy", "loudness", "speechiness",
         "acousticness", "liveness", "tempo", "duration_ms", "mode"]
ORDEM = ["Taylor Swift", "Fearless (Taylor's Version)", "Speak Now",
         "Red (Taylor's Version)", "1989", "reputation", "Lover",
         "folklore", "evermore", "Midnights"]

@st.cache_data
def carregar() -> pd.DataFrame:
    df = pd.read_csv(DADOS)
    X, y, grupos = df[AUDIO], df["valence"], df["album_name"]
    pred = np.zeros(len(df))
    for treino, teste in LeaveOneGroupOut().split(X, y, grupos):
        modelo = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        modelo.fit(X.iloc[treino], y.iloc[treino])
        pred[teste] = modelo.predict(X.iloc[teste])
    df["valence_pred"] = pred
    return df


st.set_page_config(page_title="Swiftlytics", layout="wide")
st.title("Swiftlytics")
st.caption("Taylor Swift's studio albums through Spotify audio features. "
           "191 songs; valence predicted from audio with each album held out.")

df = carregar()
album = st.sidebar.selectbox("Album", ORDEM)
sub = df[df["album_name"] == album]

c1, c2, c3 = st.columns(3)
c1.metric("Songs", len(sub))
c2.metric("Actual valence (mean)", f"{sub['valence'].mean():.2f}")
c3.metric("Predicted valence (mean)", f"{sub['valence_pred'].mean():.2f}",
          delta=f"{sub['valence_pred'].mean() - sub['valence'].mean():+.2f} vs actual",
          delta_color="off")

st.subheader("Average audio features across eras")
medias = (df.groupby("album_name")[["valence", "energy", "acousticness", "danceability"]]
            .mean().loc[ORDEM])
st.line_chart(medias)

st.subheader(f"{album}: actual vs predicted valence per song")
st.scatter_chart(sub, x="valence", y="valence_pred")

st.dataframe(
    sub[["track_number", "track_name", "valence", "valence_pred"]].round(2),
    hide_index=True,
    use_container_width=True,
)