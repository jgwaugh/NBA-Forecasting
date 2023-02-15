from typing import List, Dict, Optional

import numpy as np
from numpy.typing import NDArray


from sklearn.base import RegressorMixin

from copy import deepcopy

from sklearn.utils import check_X_y

from sklearn.model_selection import LeavePGroupsOut

from sklearn.model_selection import GridSearchCV

class ErrorPredictor(object):
    """
    Predicts non parametric standard errors

    Attributes
    ----------
    stats : List
            List of statistics names with which to build confidence intervals for
    base_model : RegressorMixin
        The base model to use to predict errors
    param_grid : Dict
        Hyperparameter grid of parameters and values to tune
    models : dict
        Dictionary mapping stats to the model predicting their standard errors
    best_model : RegressorMixin
        The model with the best hyperparameters - once this is found, all models
        use these parameters


    """

    def __init__(self,
                 stats : List,
                 base_model : RegressorMixin,
                 param_grid : Dict[str, List],
                 ):
        """

        Parameters
        ----------
        stats : List
            List of statistics names with which to build confidence intervals for
        base_model : RegressorMixin
            The base model to use to predict errors
        param_grid : Dict
            Hyperparameter grid of parameters and values to tune
        """

        self.base_model = base_model
        self.param_grid = param_grid
        self.models = {
            stat : None
            for stat in stats
        }
        self.best_model = None

    def _fit_model_hyper_params(self, X: NDArray, y: NDArray,
                        groups: NDArray ):
        """

        Tunes the hyperparameters using cross validation
        split with the `groups` parameter

        Parameters
        ----------
        X : array like
            Training data
        y : array like
            Response data
        groups : array like
            groups with which to split the cross validation

        Returns
        -------
        none
            finds the best hyperparameters

        """

        X, y = check_X_y(X, y)
        base_model = deepcopy(self.base_model)

        lpgo = LeavePGroupsOut(n_groups=1)
        split = [x for x in lpgo.split(X, y, groups)]

        model = GridSearchCV(
            estimator=base_model,
            n_jobs=3,
            cv=split,
            param_grid=self.param_grid
        )

        model.fit(X, y)

        self.best_model = model.best_estimator_

