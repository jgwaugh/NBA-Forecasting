import pickle
from os.path import abspath, dirname
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import TransformerMixin
from tensorflow.keras import Sequential

from forecasting.training.model import load_model, predict_player_career
from forecasting.training.train import filter_set


def build_stat_time_series(
    stat: str,
    player: str,
    player_df: pd.DataFrame,
    predicted_career_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds out a time series for a given statistic using predicted
    values of the statistic and raw values. Time series is in the format

    (residuals, X_{t-1}, t-1, name)

    Parameters
    ----------
    stat : str
        The statistic to predict
    player : str
        The name of the player
    player_df : pd.DataFrame
        Dataframe containing all raw statistic values for the player
    predicted_career_df : pd.DataFrame
        DataFrame containing all predicted statistic values for the player

    Returns
    -------
    pd.DataFrame
        The statistics in design matrix format

    """
    X_t = player_df.iloc[2:, :]
    pred_t = predicted_career_df.iloc[2:, :]
    X_t_minus_one = player_df.iloc[1:-1, :]

    time_minus_one = np.arange(2, len(X_t) + 2) - 1
    residual = np.log(np.abs(X_t[stat].values - pred_t[stat].values))

    return pd.DataFrame(
        {
            "log_residual": residual,
            "x": X_t_minus_one[stat].values,
            "t": time_minus_one,
            "player": [player] * len(time_minus_one),
        }
    )


def build_player_time_series_dict(
    player: str,
    players: List,
    df_raw: pd.DataFrame,
    val: List[Tuple[str, ArrayLike]],
    scaler: TransformerMixin,
    model: Sequential,
) -> Dict[str, pd.DataFrame]:
    """
    For each statistic we use to define a career, predicts the statistic using the LSTM
    and builds a time series of predicted errors and lagged values, to be used in generating
    standard errors.

    Parameters
    ----------
    player : str
        The name of the player
    players : List
        List of all player names.
    df_raw : pd.DataFrame
        pandas DataFrame of raw statistic values
    val : List
        The validation data - list of tuples of the form [(name, career)]
    scaler : TransformerMixin
        sklearn transformer used to scale between the model prediction space and the real world space
    model : Sequential
        LSTM model for predicting

    Returns
    -------
    dict
        A dictionary where keys are statistics and values are dataframes containing their standard error time
        series

    """
    stats = df_raw.columns[2:-2].tolist()

    player_idx = np.where(np.array(players) == player)[0][0]
    predicted_career = predict_player_career(val[player_idx][1], model)

    predicted_career = scaler.inverse_transform(
        predicted_career
    )  # scale back to regular space

    player_df = df_raw[df_raw.PLAYER == player]
    predicted_career_df = pd.DataFrame(
        predicted_career, columns=player_df.columns[2:-2]
    )

    return {
        stat: build_stat_time_series(stat, player, player_df, predicted_career_df)
        for stat in stats
    }


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
players = [x[0] for x in val]


###########################################################################
#
# Iterate over players and extract residuals
#
###########################################################################

stat_data = {}
for player in players:
    player_stats = build_player_time_series_dict(
        player, players, df_raw, val, scaler, model
    )
    for stat, data in player_stats.items():
        if stat not in stat_data:
            stat_data[stat] = data
        else:
            stat_data[stat] = pd.concat([stat_data[stat], data])

with open("residuals.pkl", "wb") as fp:
    pickle.dump(stat_data, fp)
