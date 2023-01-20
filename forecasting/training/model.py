import os
import warnings
from pathlib import Path
from typing import Tuple

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


import numpy as np
from forecasting.training.batchgenerator import BatchGenerator
from numpy.typing import NDArray
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import Sequential


def create_model() -> Sequential:
    """Creates the LSTM model"""

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(None, 53), batch_size=None))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(53, activation="linear")))

    return model


def load_model(
    train_generator: BatchGenerator, val_generator: BatchGenerator, retrain: bool = True
) -> Tuple:
    """
    Trains the model, or loads it from disk if it already exists

    Parameters
    ----------
    train_generator : BatchGenerator
        The training generator
    val_generator : BatchGenerator
        The validation generator
    retrain : bool
        Indicator if we should retrain or not

    Returns
    -------
    Tuple
        Training metrics and final model

    """

    model = create_model()
    no_weights_exist = not os.path.exists(
        os.path.join(Path(__file__).parents[0], "weights/weights.index")
    )

    train = retrain or no_weights_exist

    if train:
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])

        train_history = model.fit_generator(
            train_generator.generate(),
            steps_per_epoch=train_generator.data_size,
            epochs=30,
            validation_data=val_generator.generate(),
            validation_steps=val_generator.data_size,
        )

        model.save_weights(os.path.join(Path(__file__).parents[0], "weights/weights"))

        return train_history, model
    else:

        model.load_weights(f"{Path(__file__).parents[0]}/weights/weights")

        return None, model


def predict_player_career(
    career_data: NDArray,
    model: Sequential,
    n_pred: bool = 10,
    full_career: bool = True,
    look_back: int = 2,
) -> NDArray:
    """
    Predicts the career for an NBA player using a LSTM

    Parameters
    ----------
    career_data : numpy array
        The current career of the player
    model : Sequential
        LSTM RNN model
    n_pred : int
        The number of years into the future to predict
    full_career : bool
        Boolean indicating if we want to predict the same number of years as the players'
        actual career
    look_back : int
        The number of years used to start the prediction. E.g if look_back = 2, we use
        the first two years of the player's career for prediction.

    Returns
    -------
    numpy array
        Vector of career data

    """

    if full_career:
        n_pred = len(career_data) - look_back

    # begin prediction with first n seasons
    career_pred = career_data[:look_back, :]
    career_pred_tensor = np.reshape(career_pred, (1, look_back, 53))
    first_pred = model.predict(career_pred_tensor)[0, -1, :]
    career_pred = np.vstack((career_pred, first_pred))

    for i in range(n_pred - 1):
        num_seasons_pred = career_pred.shape[0]
        career_pred_tensor = np.reshape(career_pred, (1, num_seasons_pred, 53))
        prediction = model.predict(career_pred_tensor)[0, -1, :]
        career_pred = np.vstack((career_pred, prediction))

    return career_pred
