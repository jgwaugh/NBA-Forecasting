from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV, LeavePGroupsOut
from sklearn.utils import check_array, check_X_y


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

    def __init__(
        self,
        stats: List,
        base_model: RegressorMixin,
        param_grid: Dict[str, List],
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
        self.models = {stat: None for stat in stats}
        self.best_model = None

    def fit_stat_model(self, stat: str, X: NDArray, y: NDArray, groups: NDArray):
        """
        Fits the model for a specific statistic

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
        None
            Stores the best model in the class

        """
        self.models[stat] = self._fit_model(X, y, groups)

    def _fit_model(
        self, X: NDArray, y: NDArray, groups: Optional[NDArray] = None
    ) -> RegressorMixin:
        """

        Fits the model. If no hyperparameters have been tuned, tunes hyperparameters

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
        RegressorMixin
            the fitted model

        """
        X, y = check_X_y(X, y)

        if isinstance(self.best_model, type(None)):
            return self._fit_model_hyper_params(X, y, groups)
        else:
            model = deepcopy(self.best_model)
            model.fit(X, y)
            return model

    def _fit_model_hyper_params(self, X: NDArray, y: NDArray, groups: NDArray):
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
        RegressorMixin
           Regressor with the best hyperparameters

        """

        X, y = check_X_y(X, y)
        base_model = deepcopy(self.base_model)

        lpgo = LeavePGroupsOut(n_groups=1)
        split = [x for x in lpgo.split(X, y, groups)]

        model = GridSearchCV(
            estimator=base_model, n_jobs=3, cv=split, param_grid=self.param_grid
        )

        model.fit(X, y)

        self.best_model = deepcopy(model.best_estimator_)
        return deepcopy(model.best_estimator_)

    def predict_sigma(self, stat: str, X: NDArray) -> NDArray:
        """
        Predicts the standard error of a set of observations

        Parameters
        ----------
        stat : str
            The statistic to predict standard errors for
        X : array like
            The data to predict with

        Returns
        -------
        numpy array
            The standard errors for the given statistic

        """

        X = check_array(X)
        model = self.models[stat]

        if isinstance(model, type(None)):
            raise ValueError(f"Model has not yet been fitted for {stat}")

        return np.exp(model.predict(X))

    def get_model(self, stat: str):
        """Gets the model associated with the given statistic"""
        return self.models[stat]
