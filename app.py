import pickle
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from numpy.typing import NDArray

from forecasting.training.model import load_model, predict_player_career


def load_current_players(current_season: int) -> List[Tuple[str, NDArray]]:
    """
    Loads all players who played in the current NBA season

    Parameters
    ----------
    current_season : int
        The year of the season to load players from

    Returns
    -------
    list of tuples
        List of tuples where element 0 is player name and element 1 is an
        array of their career

    """

    data_directory = Path("forecasting").joinpath("munging").joinpath("data")

    with open(data_directory.joinpath("val.pkl"), "rb") as fp:
        val_load = pickle.load(fp)

    with open(data_directory.joinpath("train.pkl"), "rb") as fp:
        train_load = pickle.load(fp)

    with open(data_directory.joinpath("prediction.pkl"), "rb") as fp:
        prediction_load = pickle.load(fp)

    df = pd.read_pickle(data_directory.joinpath("NBA_stats.pkl"))

    current_players = set(df[df.YR == current_season].PLAYER.unique().tolist())

    return (
        [x for x in val_load if x[0] in current_players]
        + [y for y in train_load if y[0] in current_players]
        + [z for z in prediction_load if z[0] in current_players]
    )


###########################################################################
#
# Load in data and model
#
###########################################################################


data_directory = Path("forecasting").joinpath("munging").joinpath("data")


df_raw = pd.read_pickle(data_directory.joinpath("NBA_stats.pkl"))
scaler = joblib.load(data_directory.joinpath("scaler"))

_, model = load_model(None, None, retrain=False)

players = load_current_players(2022)


###########################################################################
#
# Application
#
###########################################################################

st.write(
    """
# Player Career Prediction Validation 

This application predicts the trajectory of an NBA player over the course of their career,
based on all available career data as of the 2022 season. 

"""
)

player_names = [x[0] for x in players]
stats = df_raw.columns[2:-2].tolist()

player = st.selectbox("Select a player to view forecasts of their career", player_names)
n_seasons_predict = st.slider(
    "Use this slider to set the number of seasons to predict", 1, 10, 3
)


player_idx = np.where(np.array(player_names) == player)[0][0]
len_current_career = len(players[player_idx][1])

predicted_career = predict_player_career(
    players[player_idx][1],
    model,
    full_career=False,
    n_pred=n_seasons_predict,
    look_back=len_current_career,
)

predicted_career = scaler.inverse_transform(
    predicted_career
)  # scale back to regular space

stat = st.selectbox("Select a stat of forecasts to view", stats)

player_df = df_raw[df_raw.PLAYER == player]
prediction_df = player_df.copy()

years = player_df.YR.values.tolist()
last_year = years[-1]
years += [last_year + j for j in range(1, n_seasons_predict + 1)]

predicted_career_df = pd.DataFrame(predicted_career, columns=player_df.columns[2:-2])
predicted_career_df.insert(0, "YR", years)
predicted_career_df.insert(0, "PLAYER", [player] * len(predicted_career_df))

true_stat = player_df[stat].values
predicted_stat = predicted_career_df[stat].values
yrs_actual = player_df.YR.values


fig = plt.figure()

plt.plot(years, predicted_stat, color="red", label="Forecast")
plt.plot(yrs_actual, true_stat, color="blue", label="Actual Career")
plt.ylabel(stat)
plt.xlabel("Year")
plt.title(player + " Career " + stat + " and Predictions")
plt.legend(loc="upper left")
plt.xticks(rotation=30)

st.pyplot(fig)
