# Swiftlytics

A data science project that explores Taylor Swift's discography through Spotify audio features. It analyzes how the sound of each album changes across eras and tests how much of a song's mood (valence) can be predicted from audio alone. The dataset has 191 songs from studio albums, so results should be read as exploratory.

## Features

- **Data cleaning**: removes tracks without audio features and fixes column types (`src/limpeza.py`).
- **Exploratory analysis**: average valence, energy, acousticness and danceability per album in era order, plus a correlation heatmap (`src/eda.py`).
- **Valence model**: compares a mean baseline, Ridge regression and Random Forest with 5-fold cross-validation (`src/modelo_valencia.py`).
- **Album-level validation**: holds out entire albums to avoid optimistic scores and reports Ridge coefficients (`src/avaliacao.py`).
- **Dashboard**: Streamlit app to browse albums and compare actual vs predicted valence per song (`streamlit run src/dashboard.py`).

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn

## Getting Started

1. Clone the repository and enter the folder.
2. Create and activate a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
3. Install the dependencies: `pip install -r requirements.txt`
4. Download the dataset: `wget -P data/raw https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2023/2023-10-17/taylor_album_songs.csv`
5. Run the scripts in order: `python src/limpeza.py`, `python src/eda.py`, `python src/modelo_valencia.py`, `python src/avaliacao.py`

## Results

- Audio features alone explain about a third of a song's valence (Ridge, R² 0.33, held-out albums, 191 songs).
- On the 130 songs with lyrics (8 albums), adding simple lyric features (VADER sentiment, vocabulary diversity, line repetition) did not improve predictions: audio only R² 0.21 vs audio + text R² 0.10.
- Interpret with care: small sample, folds with only 1-2 albums, and the lyrics dataset does not cover *evermore* or *Midnights*.
- On the 130 songs with lyrics (8 albums), lyric features did not improve predictions. Simple features (VADER sentiment, vocabulary diversity, line repetition) and TF-IDF (300 terms, three Ridge penalties) both matched or underperformed audio only (R² 0.21-0.24).
- Leaving one album out at a time, the audio model beat the mean baseline on 9 of 10 albums (mean absolute error 0.125 vs 0.157). It overestimated the mood of *reputation* and *Midnights*, the two lowest-valence albums (predicted ~0.42 and ~0.39 vs actual ~0.29 and ~0.28), which suggests danceability and energy can mask sad-sounding songs.

![Real vs predicted valence per album](reports/figures/real_vs_previsto_por_album.png)

## Author

Laura Vieira - [@LaurVieira](https://github.com/LaurVieira)