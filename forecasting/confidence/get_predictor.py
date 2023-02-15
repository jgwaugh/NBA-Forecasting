import pickle

import numpy as np

from forecasting.confidence import ErrorPredictor

from pathlib import Path

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from forecasting.confidence import ErrorPredictor

def get_predictor_model() -> ErrorPredictor:
    """Loads the error predictor - if it has not been fit, trains it """

    model_path = Path(__file__).parents[0].joinpath("sigma_predictor.pkl")

    try:
        with open(model_path, "rb") as fp:
           return pickle.load(fp)
    except FileNotFoundError:
        with open(Path(__file__).parents[0].joinpath('residuals.pkl'), "rb") as fp:
            stats = pickle.load(fp)

        stat_names = [k for k, v in stats.items()]

        reg = GradientBoostingRegressor(
            max_depth=3,
            n_estimators=100
        )

        param_grid = {
            'max_depth': [2, 3, 5],
            'n_estimators': [25, 50, 75, 100]
        }

        group_map = None
        n_groups = 5

        predictor = ErrorPredictor(
            base_model=reg,
            stats=stat_names,
            param_grid=param_grid
        )

        for stat, data in stats.items():
            if isinstance(group_map, type(None)):
                players = data.player.unique()
                np.random.shuffle(players)
                group = np.arange(len(players)) % n_groups

                player_groups = pd.DataFrame({
                    'player': players,
                    'group': group
                })
                group_map = player_groups

            data = data.merge(group_map, how='left')

            groups = data.group.values

            X = data[['x', 't']]
            y = data['log_residual']

            predictor.fit_stat_model(
                stat,
                X,
                y, groups
            )

        with open(model_path, "wb") as fp:
            pickle.dump(predictor, fp)
        return predictor



