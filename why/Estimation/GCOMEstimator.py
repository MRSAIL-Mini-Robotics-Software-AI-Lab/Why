"""
A GCOM (Grouped Conditional Outcome Modeling) estimator
"""
from typing import List

import numpy as np
import pandas as pd

from .Estimator import Estimator  # pylint: disable=no-name-in-module


class GCOMEstimator:
    """
    A GCOM (Grouped Conditional Outcome Modeling) estimator that can be used with any model
    that implements the interface (Protocol) of the Estimator class
    """

    def __init__(self, data_frame: pd.DataFrame, model: Estimator):
        self.data_frame = data_frame
        self.model = model

    def estimate(
        self, treatment: str, outcome: str, adjustment_set: List[str]
    ) -> float:
        """
        Estimate the ATE of a treatment on an outcome using a GCOM
        (Grouped Conditional Outcome Modeling) estimator.

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
        if len(adjustment_set) == 0:
            print(
                "Warning, GCOM: No adjustment set provided, returning ate = E[Y|T=1]-E[Y|T=0]"
            )
            treatment = self.data_frame[treatment].values
            outcome = self.data_frame[outcome].values
            outcome_0 = (
                np.sum(outcome[treatment == 0]) / outcome[treatment == 0].shape[0]
            )
            outcome_1 = (
                np.sum(outcome[treatment == 1]) / outcome[treatment == 1].shape[0]
            )
            return outcome_1 - outcome_0

        feature_cols = adjustment_set
        target_col = outcome

        features = self.data_frame[feature_cols].values
        treatment = self.data_frame[treatment].values
        target = self.data_frame[target_col].values

        # Model (T==0)
        features_0 = features[treatment == 0]
        targets_0 = target[treatment == 0]
        self.model.fit(features_0, targets_0)

        preds_t0 = self.model.predict(features)

        # Model (T==1)
        features_1 = features[treatment == 1]
        targets_1 = target[treatment == 1]
        self.model.fit(features_1, targets_1)
        preds_t1 = self.model.predict(features)

        ate = np.mean(preds_t1 - preds_t0)

        return ate
