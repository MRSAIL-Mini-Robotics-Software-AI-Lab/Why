from typing import List

import numpy as np
import pandas as pd


class LinearRegressionATE:
    def __init__(self, data_frame:pd.DataFrame):
        self.data_frame = data_frame

    def estimate(self, treatment:str, outcome:str, adjustment_set:List[str]) -> float:
        feature_cols = [treatment] + adjustment_set
        target_col = outcome

        features = self.data_frame[feature_cols].values
        target = self.data_frame[target_col].values

        features = np.hstack((features, np.ones((features.shape[0], 1))))

        theta = np.linalg.pinv(features) @ target

        ate = theta[0]

        return ate
