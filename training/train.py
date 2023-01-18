import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from os.path import abspath, dirname
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
from batchgenerator import BatchGenerator
from numpy.typing import NDArray
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, TimeDistributed
from tensorflow.keras.models import Sequential


def filter_set(
    data: List[Tuple[str, NDArray]], num_seasons: int
) -> List[Tuple[str, NDArray]]:
    """
    Filters the data so that only players with a minimum number of seasons played are included

    Parameters
    ----------
    data : list of tuples
        List of tuples of the form (player_name, season)
    num_seasons : int
        Minimum number of seasons for a player

    Returns
    -------
    list of tuples
        The list of players, with players having shorter seasons filtered out

    """

    return [x for x in data if x[1].shape[0] >= num_seasons]


if __name__ == "__main__":
    data_directory = (
        Path(dirname(dirname(abspath(__file__)))).joinpath("munging").joinpath("data")
    )

    with open(data_directory.joinpath("val.pkl"), "rb") as fp:
        val_load = pickle.load(fp)

    with open(data_directory.joinpath("train.pkl"), "rb") as fp:
        train_load = pickle.load(fp)

    train = filter_set(train_load, 3)
    val = filter_set(val_load, 3)

    train_generator = BatchGenerator(train)
    val_generator = BatchGenerator(val)
