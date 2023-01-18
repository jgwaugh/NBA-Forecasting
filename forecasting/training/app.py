import pickle
from os.path import abspath, dirname
from pathlib import Path

import ipdb
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from batchgenerator import BatchGenerator
from model import load_model, predict_player_career
from nba_forecasting.munging.load_data import stats
from train import filter_set

ipdb.set_trace()

data_directory = (
    Path(dirname(dirname(abspath(__file__)))).joinpath("munging").joinpath("data")
)

with open(data_directory.joinpath("val.pkl"), "rb") as fp:
    val_load = pickle.load(fp)

val = filter_set(val_load, 3)

df_raw = pd.read_pickle(data_directory.joinpath("NBA_stats.pkl"))
df_transformed = pd.read_pickle(data_directory.joinpath("NBA_stats_transformed.pkl"))
scaler = joblib.load(data_directory.joinpath("scaler"))

_, model = load_model(None, None, refit=False)


st.write(
    """
# Player Career Prediction Validation
Below, we can view predictions of various player careers, along with their actual
careers, as a way of validating model performance. 
"""
)

players = [x[0] for x in val]
player = st.selectbox("Select a player to view forecasts of their career", players)

player_idx = np.where(np.array(players) == player)[0][0]

predicted_career = predict_player_career(val[player_idx][1], model)
