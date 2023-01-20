import pickle
from os.path import abspath, dirname
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from model import load_model
from numpy.typing import NDArray
from tensorflow.keras.models import Sequential
from train import filter_set


def predict_career_steps(career_data: NDArray, model: Sequential) -> NDArray:
    """Predicts X_{t+ 1} given all X_t, X_{t-1}, .., X_t"""
    N = len(career_data)
    predictions = []

    for t in range(2, N):
        career_to_t = career_data[:t, :].reshape(1, -1, 53)
        prediction = model.predict(career_to_t)[0, -1, :]
        predictions.append(prediction.reshape(1, -1))

    return np.vstack(predictions)


def generate_baseline_comparison(
    data: List[Tuple[str, NDArray]], model: Sequential
) -> Tuple[float]:
    """
    Compares the model performance to a lagged baseline. Here, error is defined
    as the L2 distance between a predicted season and the actual season,
    averaged over all 53 statistics. That error is then averaged over a player's career
    to get a per player measure of error, which is then averaged over players to get a per model error,
    which is what is returned.

    Parameters
    ----------
    data : list
        List of tuples (name, career)
    model : Sequential
        LSTM predictive model

    Returns
    -------
    tuple
        tuple of floats containing the baseline error and the model's error

    """

    model_error = []
    baseline_error = []

    for player_history in data:
        career_prediction = predict_career_steps(player_history[1], model)
        true_career = player_history[1][2:, :]
        lagged_career = player_history[1][1:-1, :]

        lagged_error = (lagged_career - true_career) ** 2
        pred_error = (career_prediction - true_career) ** 2

        lagged_error = lagged_error.mean(axis=1).mean()
        pred_error = pred_error.mean(axis=1).mean()

        model_error.append(pred_error)
        baseline_error.append(lagged_error)

    model_error = np.mean(model_error)
    baseline_error = np.mean(baseline_error)

    return model_error, baseline_error


data_directory = (
    Path(dirname(dirname(abspath(__file__)))).joinpath("munging").joinpath("data")
)

with open(data_directory.joinpath("val.pkl"), "rb") as fp:
    val_load = pickle.load(fp)

val = filter_set(val_load, 3)

_, model = load_model(None, None, retrain=False)

model_error, baseline_error = generate_baseline_comparison(val, model)

df_plt = pd.DataFrame(
    {"Model": ["Lag", "LSTM"], "Error": [baseline_error, model_error]}
)

f = plt.figure(dpi=100)

sns.barplot(x="Model", y="Error", data=df_plt)

f.savefig("images/baseline_error_comparison.png")
