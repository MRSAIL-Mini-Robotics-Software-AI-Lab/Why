"""
Estimator Protocol
"""
from typing import Protocol
import numpy as np


class Estimator(Protocol):
    """
    Estimator Protocol
    """

    def fit(self, features: np.array, target: np.array) -> None:
        """
        Fit the model to the data

        Parameters
        ----------
        features : np.array
            Features as input to the model
        target : np.array
            Target as output of the model
        """

    def predict(self, features: np.array) -> np.array:
        """
        Predict the target given the features

        Parameters
        ----------
        features : np.array
            Features as input to the model

        Returns
        -------
        np.array
            Predicted target
        """
