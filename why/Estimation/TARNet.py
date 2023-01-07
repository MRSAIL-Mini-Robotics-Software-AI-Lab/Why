"""
TARNet Estimator
"""
from typing import List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


class TARModel(nn.Module):
    """
    Torch model for TARNet
    Contains the model body and two heads for treatment 0 and 1

    Parameters
    ----------
    input_size : int
        Number of features as input
    hidden_size : int
        Number of hidden units in the model body
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.model_body = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())

        self.model_head_0 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.model_head_1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, features: torch.tensor, treatment: torch.tensor):
        """
        Forward pass of the model

        Parameters
        ----------
        features : torch.tensor
            Tensor of features, shape = (batch_size, input_size)
        treatment : torch.tensor
            Tensor of treatment, shape = (batch_size, 1)

        Returns
        -------
        torch.tensor
            Tensor of predictions, shape = (batch_size, 1)
        """
        body = self.model_body(features)
        pred_0 = self.model_head_0(body)
        pred_1 = self.model_head_1(body)

        pred = pred_0[0] * (1 - treatment) + pred_1[0] * treatment
        return pred


class TARNet:
    """
    TARNet Estimator for estimating ATE

    Parameters
    ----------
    data_frame: pd.DataFrame
        Data frame containing the data
    """

    def __init__(self, data_frame: pd.DataFrame):
        self.data_frame = data_frame

    def estimate(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: List[str],
        n_epochs: int = 200,
        learning_rate: float = 0.001,
        hidden_size: int = 100,
    ) -> float:
        """
        Estimate the ATE using TARNet

        Parameters
        ----------
        treatment : str
            Variable name of the treatment
        outcome : str
            Variable name of the outcome
        adjustment_set : List[str]
            List of variable names to adjust for
        n_epochs : int, by default 200
            Number of epochs to train for
        learning_rate : float, by default 0.001
            Learning rate for the optimizer
        hidden_size : int, by default 100
            Number of hidden units in the model body

        Returns
        -------
        float
            Estimated ATE
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

        features = torch.from_numpy(self.data_frame[feature_cols].values).float()
        treatment = torch.from_numpy(self.data_frame[treatment].values).float()
        target = torch.from_numpy(self.data_frame[target_col].values).float()

        model = TARModel(features.shape[1], hidden_size)
        model = self.fit(features, treatment, target, model, n_epochs, learning_rate)

        preds_t0 = model.forward(features, torch.zeros(features.shape[0], 1))  # T=0
        preds_t1 = model.forward(features, torch.ones(features.shape[0], 1))  # T=1

        ate = torch.mean(preds_t1 - preds_t0).detach().numpy()

        return ate

    def fit(
        self,
        features: torch.tensor,
        treatment: torch.tensor,
        target: torch.tensor,
        model: nn.Module,
        n_epochs: int,
        learning_rate: float,
    ) -> nn.Module:
        """
        Fit the model

        Parameters
        ----------
        features : torch.tensor
            Features as input to the model
        treatment : torch.tensor
            Treatment as input to the model
        target : torch.tensor
            Target as output of the model
        model : nn.Module
            Model to fit
        n_epochs : int
            Number of epochs to train for
        learning_rate : float
            Learning rate for the optimizer

        Returns
        -------
        nn.Module
            Fitted model
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for _ in range(n_epochs):
            pred = model.forward(features, treatment)

            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model
