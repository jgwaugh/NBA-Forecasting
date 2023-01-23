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
    current_idx : int
        The current player index as the generator moves sequentially - once it reaches
        the max size, it resets to zero
    data_size : int
        The size of the data set

    """

    def __init__(self, data: List[Tuple[str, NDArray]]):
        """
        Parameters
        ----------
        data : list
            list of tuples of the form (player_name, career)

        """
        self.data = data
        self.current_idx = 0
        self.data_size = len(data)

    def generate(self):
        """Yields prediction pairs of the form X =(X_{i-step_size},..., X_{i - 1})
        and y = X_i"""
        while True:
            if self.current_idx >= len(self.data):
                self.current_idx = 0

            player = self.data[self.current_idx][1]
            num_obs = player.shape[0]

            if num_obs == 2:
                step_size = 1
            elif num_obs < 9:
                step_size = 3
            else:
                step_size = 4

            batchsize = player.shape[0] - step_size

            X = np.zeros((batchsize, step_size, player.shape[1]))
            y = np.zeros((batchsize, step_size, player.shape[1]))

            for i in range(0, batchsize):
                X[i, :, :] = player[i : i + step_size, :]
                y[i, :, :] = player[i + 1 : i + 1 + step_size, :]
            self.current_idx += 1

            yield X, y
