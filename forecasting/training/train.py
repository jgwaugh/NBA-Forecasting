import pickle
import warnings
from os.path import abspath, dirname
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from forecasting.training.model import load_model

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from numpy.typing import NDArray

from forecasting.training.batchgenerator import BatchGenerator


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

    train_history, model = load_model(train_generator, val_generator, retrain=True)

    train_loss = train_history.history["loss"]
    val_loss = train_history.history["val_loss"]
    epochs = range(1, 31)

    fig = plt.figure(dpi=100)

    plt.plot(epochs, val_loss, color="blue", label="Validation Loss")

    plt.ylabel("Least Squares Loss")
    plt.xlabel("Epoch")
    plt.title("Validation Loss vs Epoch")
    plt.show()

    fig.savefig("images/validation_loss.png", bbox_inches="tight")
