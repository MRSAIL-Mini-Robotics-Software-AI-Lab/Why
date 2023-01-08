"""
Generative Neural Network for estimating the direction of an edge in a structural causal model
Based on the CGNNs paper, https://arxiv.org/abs/1711.08936
"""
import copy
from tqdm import tqdm

import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.nn as nn

from .utils import MMDLoss  # pylint: disable=no-name-in-module


class GNN(nn.Module):
    """
    Generative Neural Network for estimating a variable from its causes

    Parameters
    ----------
    n_inputs : int
        Number of parent variables of the variable to be estimated
    hidden_size: int
        Number of hidden units in the model
    """

    def __init__(self, n_inputs: int, hidden_size: int):
        super().__init__()
        self.n_inputs = n_inputs
        self.model = nn.Sequential(
            nn.Linear(n_inputs + 1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.tensor
            Tensor of features, shape = (batch_size, n_inputs)

        Returns
        -------
        torch.tensor
            Tensor of predictions, shape = (batch_size, 1)
        """
        device = next(self.parameters()).device
        x = x.reshape(-1, self.n_inputs)
        noise = torch.normal(0, 1, (x.shape[0], 1)).to(device)

        inp = torch.cat([x, noise], axis=1)
        return self.model.forward(inp)


class GNNOrientation:
    """
    Class for estimating the causal direction of an edge in a structural causal model
    using a Generative Neural Network
    """

    def __init__(
        self,
        hidden_size: int = 5,
        n_orientation_runs: int = 6,
        n_train_runs: int = 300,
        n_eval_runs: int = 100,
        learning_rate: float = 1e-2,
        batch_size: int = -1,
    ):
        self.hidden_size = hidden_size
        self.n_orientation_runs = n_orientation_runs
        self.n_train_runs = n_train_runs
        self.n_eval_runs = n_eval_runs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def eval_gnn(
        self,
        cause: torch.tensor,
        effect: torch.tensor,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Trains a GNN and returns the evaluated MMD loss after training

        Parameters
        ----------
        cause : torch.tensor
            Tensor of variable samples, shape = (batch_size, 1)
            This is the variable assumed to be the cause of `effect`
        effect : torch.tensor
            Tensor of variable samples, shape = (batch_size, 1)
            This is the variable assumed to be the effect of `cause`
        device: torch.device, by default cpu
            Device to run the model on

        Returns
        -------
        float
            Mean MMD loss over all evaluations
        """
        batch_size = self.batch_size
        if self.batch_size == -1:
            batch_size = cause.shape[0]
        model = GNN(1, self.hidden_size)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        target = torch.cat([cause, effect], dim=1)

        # Train the model
        for _ in range(self.n_train_runs):
            indices = torch.randint(0, cause.shape[0], (batch_size,))
            pred = model.forward(cause[indices])
            pred = torch.cat([cause[indices], pred], dim=1)

            loss = MMDLoss(target[indices], pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model
        losses = []
        with torch.no_grad():
            for _ in range(self.n_eval_runs):
                indices = torch.randint(0, cause.shape[0], (batch_size,))
                pred = model.forward(cause[indices])
                pred = torch.cat([cause[indices], pred], dim=1)

                loss = MMDLoss(target[indices], pred)
                losses.append(loss.item())

        return np.mean(losses)

    def orient_edge(
        self,
        var_x,
        var_y,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
    ):
        """
        Estimates the causal direction of an edge

        Parameters
        ----------
        var_x : torch.tensor
            Tensor of variable samples, shape = (batch_size, 1)
        var_y : torch.tensor
            Tensor of variable samples, shape = (batch_size, 1)
        device: torch.device, by default cpu
            Device to run the model on
        verbose : bool, by default False
            If True, prints a progress bar

        Returns
        -------
        float
            If > 0, then `var_x` is the cause of `var_y`
                If < 0, then `var_y` is the cause of `var_x`
            If == 0, then the causal direction cannot be determined (no causal relationship)
        """
        losses_xy = []
        losses_yx = []
        for _ in tqdm(range(self.n_orientation_runs), disable=not verbose):
            losses_xy.append(self.eval_gnn(var_x, var_y, device))
            losses_yx.append(self.eval_gnn(var_y, var_x, device))
        mean_xy = np.mean(losses_xy)
        mean_yx = np.mean(losses_yx)
        causal_dir_score = (mean_yx - mean_xy) / (mean_yx + mean_xy)
        return causal_dir_score

    def orient_edges(
        self,
        graph: nx.DiGraph,
        data_frame: pd.DataFrame,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
    ):
        """
        Estimates the causal direction of all edges in a graph

        Parameters
        ----------
        graph : nx.DiGraph
            Graph to orient
        data_frame : pd.DataFrame
            Data frame containing the data to use for orientation
        device: torch.device, by default cpu
            Device to run the model on
        verbose : bool, by default False
            If True, prints a progress bar

        Returns
        -------
        nx.DiGraph
            Oriented graph
        """
        graph = copy.deepcopy(graph)
        edges = list(graph.edges)
        done = False
        while not done:
            done = True
            for edge in edges:
                new_edge = edge[1], edge[0]
                if new_edge in edges:
                    done = False
                    x = (
                        torch.from_numpy(data_frame[edge[0]].values)
                        .float()
                        .to(device)
                        .reshape(-1, 1)
                    )
                    y = (
                        torch.from_numpy(data_frame[edge[1]].values)
                        .float()
                        .to(device)
                        .reshape(-1, 1)
                    )
                    val = self.orient_edge(x, y, device, verbose)
                    assert val != 0, "Shouldn't be 0"
                    if val > 0:
                        graph.remove_edge(edge[1], edge[0])
                    elif val < 0:
                        graph.remove_edge(edge[0], edge[1])
                    edges = list(graph.edges())
                    break

        return graph
