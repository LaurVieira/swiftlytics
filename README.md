# Swiftlytics

A data science project that explores Taylor Swift's discography through Spotify audio features. It analyzes how the sound of each album changes across eras and tests how much of a song's mood (valence) can be predicted from audio alone. The dataset has 191 songs from studio albums, so results should be read as exploratory.

## Features

- **Data cleaning**: removes tracks without audio features and fixes column types (`src/limpeza.py`).
- **Exploratory analysis**: average valence, energy, acousticness and danceability per album in era order, plus a correlation heatmap (`src/eda.py`).
- **Valence model**: compares a mean baseline, Ridge regression and Random Forest with 5-fold cross-validation (`src/modelo_valencia.py`).
- **Album-level validation**: holds out entire albums to avoid optimistic scores and reports Ridge coefficients (`src/avaliacao.py`).

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

## Author

Laura Vieira - [@LaurVieira](https://github.com/LaurVieira)