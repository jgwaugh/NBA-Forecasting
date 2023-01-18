import os
import warnings
from pathlib import Path
from typing import Tuple

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


from batchgenerator import BatchGenerator
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import Sequential


def create_model() -> Sequential:
    """Creates the LSTM model"""

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(None, 57), batch_size=None))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(57, activation="linear")))

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
