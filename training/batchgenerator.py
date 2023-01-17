from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


class BatchGenerator(object):
    """
    Generator that yields covariate / prediction pairs

    Attributes
    ----------
    data : list
        list of tuples of the form (player_name, career)
    step_size : int
        The number of timesteps back to look when predicting. Set to 2 to maximize
        quantity of data.
    current_idx : int
        The current player index as the generator moves sequentially - once it reaches
        the max size, it resets to zero
    data_size : int
        The size of the data set

    """

    def __init__(self, data: List[Tuple[str, NDArray]], step_size: int = 2):
        """
        Parameters
        ----------
        data : list
            list of tuples of the form (player_name, career)
        step_size : int
            The number of timesteps back to look when predicting. Set to 2 to maximize
            quantity of data.

        """
        self.data = data
        self.current_idx = 0
        self.step_size = step_size
        self.data_size = len(data)

    def generate(self):
        """Yields prediction pairs of the form train =(X_{i-step_size},..., X_{i - 1})
        and test = X_i"""
        while True:
            if self.current_idx >= len(self.data):
                self.current_idx = 0

            player = self.data[self.current_idx][1]
            num_obs = player.shape[0]

            if num_obs == 2:
                step_size = 1
            else:
                step_size = self.step_size

            batchsize = player.shape[0] - step_size

            X = np.zeros((batchsize, step_size, player.shape[1]))
            y = np.zeros((batchsize, 1, player.shape[1]))

            for i in range(0, batchsize):
                X[i, :, :] = player[i : i + step_size, :]
                y[i, :, :] = player[i + step_size : i + 1 + step_size, :]
            self.current_idx += 1

            yield X, y
