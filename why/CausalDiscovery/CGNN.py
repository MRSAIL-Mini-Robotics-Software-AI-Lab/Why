"""
A class to improve the directions of edges in a given estimated structural causal model using
CGNNs, https://arxiv.org/abs/1711.08936
"""
import copy
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

from .utils import MMDLoss  # pylint: disable=no-name-in-module


class CGNN(nn.Module):
    """
    Causal Generative Neural Network for learning a distribution from data and causal structure

    Parameters
    ----------
    graph: nx.DiGraph
        The causal graph of the distribution
    hidden_size: int
        Number of hidden units in the model
    """

    def __init__(self, graph: nx.DiGraph, hidden_size: int = 5):
        super().__init__()
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.edges = list(graph.edges)
        self.order = list(nx.topological_sort(graph))
        self.parents = []
        self.parent_indices = []
        self.models = []
        for node in self.order:
            node_parents = list(graph.predecessors(node))
            self.parents.append(node_parents)
            parent_idx = []
            for parent in node_parents:
                parent_idx.append(self.nodes.index(parent))
            self.parent_indices.append(torch.tensor(parent_idx).long())
            model = nn.Sequential(
                nn.Linear(len(node_parents) + 1, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
            self.models.append(model)

        self._models = nn.Sequential(*self.models)  # workaround to easily use cuda

    def forward(self, n_samples: int = 100):
        """
        Generate n samples from the underlying distribution
        """
        device = next(self.parameters()).device
        generated = torch.zeros((n_samples, len(self.nodes))).to(device)
        noise = torch.normal(0, 1, generated.shape).to(device)

        for i in range(len(self.nodes)):
            inps = generated[:, self.parent_indices[i]]
            inps = torch.cat([inps, noise[:, i : i + 1]], dim=1)
            pred = self.models[i].forward(inps)
            generated[:, i : i + 1] = pred

        return generated


class CGNNOrientation:
    """
    A class to improve the directions of edges in a given estimated structural causal model and
    data using a Causal Generative Neural Network

    Parameters
    ----------
    hidden_size: int
        Number of hidden units in the model
    learning_rate: float
        Learning rate for the optimizer
    n_train_runs: int
        Number of training runs for the model
    n_eval_runs: int
        Number of evaluation runs for the model
    n_orientation_runs: int
        Number of orientation runs for the model
    batch_size: int
        Batch size for training the model
    """

    def __init__(
        self,
        hidden_size: int = 5,
        learning_rate: float = 1e-2,
        n_train_runs: int = 300,
        n_eval_runs: int = 100,
        n_orientation_runs: int = 6,
        batch_size: int = -1,
    ):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_train_runs = n_train_runs
        self.n_eval_runs = n_eval_runs
        self.n_orientation_runs = n_orientation_runs
        self.batch_size = batch_size

    def eval_graph_once(
        self,
        graph: nx.DiGraph,
        data: torch.tensor,
        device: torch.device = torch.device("cpu"),
    ) -> float:
        """
        Evaluate the graph by training a CGNN and evaluating it on the data
        using the MMD loss

        Parameters
        ----------
        graph : nx.DiGraph
            The proposed structural causal model
        data : torch.tensor
            The data sampled from the correct structural causal model
        device: torch.device, by default cpu
            Device to run the model on

        Returns
        -------
        float
            The MMD loss of the trained CGNN
        """
        batch_size = self.batch_size
        if self.batch_size == -1:
            batch_size = data.shape[0]
        model = CGNN(graph, self.hidden_size)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        for _ in range(self.n_orientation_runs):
            indices = torch.randint(0, data.shape[0], (batch_size,))
            pred = model.forward(batch_size)
            loss = MMDLoss(data[indices], pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses = []
        with torch.no_grad():
            for _ in range(self.n_eval_runs):
                pred = model.forward(batch_size)
                loss = MMDLoss(data[indices], pred)
                losses.append(loss.item())
        return np.mean(losses)

    def eval_graph(
        self,
        graph: nx.DiGraph,
        data: torch.tensor,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
    ) -> float:
        """
        Evaluate the graph by training a CGNN and evaluating it on the data
        This is done multiple times and the average loss is returned
        To evaluate only once, use eval_graph_once

        Parameters
        ----------
        graph : nx.DiGraph
            The proposed structural causal model
        data : torch.tensor
            The data sampled from the correct structural causal model
        device: torch.device, by default cpu
            Device to run the model on
        verbose : bool, by default False
            If True, show a progress bar

        Returns
        -------
        float
            The average MMD loss of the trained CGNN
        """
        losses = []
        for _ in tqdm(range(self.n_orientation_runs), disable=not verbose):
            losses.append(self.eval_graph_once(graph, data, device))
        return np.mean(losses)

    def optimize_graph(
        self,
        graph: nx.DiGraph,
        data: torch.tensor,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
    ) -> nx.DiGraph:
        """
        Optimize a given graph by iteratively flipping edges and evaluating the graph

        Parameters
        ----------
        graph : nx.DiGraph
            The starting structural causal model
        data : torch.tensor
            The data sampled from the correct structural causal model
        device: torch.device, by default cpu
            Device to run the model on
        verbose : bool, by default False
            If True, show a progress bar and prints when a better graph is found

        Returns
        -------
        nx.DiGraph
            The best graph found
        """
        visited_graphs = [list(graph.adjacency())]

        current_eval = self.eval_graph(graph, data, device, verbose)
        local_optimum = False
        while not local_optimum:
            local_optimum = True
            for edge in graph.edges():
                new_graph = copy.deepcopy(graph)
                new_graph.add_edge(edge[1], edge[0])
                new_graph.remove_edge(edge[0], edge[1])
                adj = list(new_graph.adjacency())
                if adj not in visited_graphs and nx.is_directed_acyclic_graph(
                    new_graph
                ):
                    visited_graphs.append(adj)

                    evaluation = self.eval_graph(new_graph, data, device, verbose)
                    if evaluation < current_eval:
                        if verbose:
                            print("Found better graph, switching edge ", edge)
                        graph = new_graph
                        current_eval = evaluation
                        local_optimum = False
                        break

        return graph
