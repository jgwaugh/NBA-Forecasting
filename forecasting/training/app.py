import pickle
from os.path import abspath, dirname
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from forecasting.training.model import load_model, predict_player_career
from forecasting.training.train import filter_set

###########################################################################
#
# Load in data and model
#
###########################################################################

data_directory = (
    Path(dirname(dirname(abspath(__file__)))).joinpath("munging").joinpath("data")
)

with open(data_directory.joinpath("val.pkl"), "rb") as fp:
    val_load = pickle.load(fp)

val = filter_set(val_load, 3)

df_raw = pd.read_pickle(data_directory.joinpath("NBA_stats.pkl"))
df_transformed = pd.read_pickle(data_directory.joinpath("NBA_stats_transformed.pkl"))
scaler = joblib.load(data_directory.joinpath("scaler"))

_, model = load_model(None, None, retrain=False)


###########################################################################
#
# Application
#
###########################################################################

st.write(
    """
# Player Career Prediction Validation at t = 2 years
Below, we can view predictions of various player careers, along with their actual
careers, as a way of validating model performance. 

These forecasts are made from 2 years into a player's career. The model was not trained
on these players. 

"""
)

players = [x[0] for x in val]
stats = df_raw.columns[2:-2].tolist()

player = st.selectbox("Select a player to view forecasts of their career", players)


player_idx = np.where(np.array(players) == player)[0][0]
predicted_career = predict_player_career(val[player_idx][1], model)

predicted_career = scaler.inverse_transform(
    predicted_career
)  # scale back to regular space

stat = st.selectbox("Select a stat of forecasts to view", stats)

player_df = df_raw[df_raw.PLAYER == player]
prediction_df = player_df.copy()


###########################################################################
#
# Plot Career Trajectory
#
###########################################################################

predicted_career_df = pd.DataFrame(predicted_career, columns=player_df.columns[2:-2])
predicted_career_df.insert(0, "YR", player_df["YR"].values)
predicted_career_df.insert(0, "PLAYER", player_df["PLAYER"].values)

true_stat = player_df[stat].values
predicted_stat = predicted_career_df[stat].values
yrs = player_df.YR.values


fig = plt.figure()

plt.plot(yrs, predicted_stat, color="red", label="prediction")
plt.plot(yrs, true_stat, color="blue", label="truth")
plt.ylabel(stat)
plt.xlabel("Year")
plt.title(
    player + " Career " + stat + " and Predictions \n based on " + str(2) + " seasons"
)
plt.legend(loc="upper left")
plt.xticks(rotation=30)

st.pyplot(fig)
