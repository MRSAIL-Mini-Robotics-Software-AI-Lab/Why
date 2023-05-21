"""
A COM (Conditional Outcome Modeling) estimator
"""
from typing import List

import numpy as np
import pandas as pd

from .Estimator import Estimator  # pylint: disable=no-name-in-module


class COMEstimator:
    """
    A COM (Conditional Outcome Modeling) estimator that can be used with any model
    that implements the interface (Protocol) of the Estimator class
    """

    def __init__(self, data_frame: pd.DataFrame, model: Estimator):
        self.data_frame = data_frame
        self.model = model

    def estimate(
        self, treatment: str, outcome: str, adjustment_set: List[str]
    ) -> float:
        """
        Estimate the ATE of a treatment on an outcome using a COM
        (Conditional Outcome Modeling) estimator.

        Parameters
        ----------
        treatment : str
            Variable name of the treatment
        outcome : str
            Variable name of the outcome
        adjustment_set : List[str]
            List of variable names of a sufficient adjustment set

        Returns
        -------
        float
            The estimated ATE
        """
        feature_cols = [treatment] + adjustment_set
        target_col = outcome

        features = self.data_frame[feature_cols].values
        target = self.data_frame[target_col].values

        self.model.fit(features, target)

        treatment_0_data = np.hstack(
            (np.zeros((features.shape[0], 1)), self.data_frame[adjustment_set].values)
        )
        treatment_1_data = np.hstack(
            (np.ones((features.shape[0], 1)), self.data_frame[adjustment_set].values)
        )
        preds_t0 = self.model.predict(treatment_0_data)
        preds_t1 = self.model.predict(treatment_1_data)

        ate = np.mean(preds_t1 - preds_t0)

        return ate
