"""
Orients the edges of a DAG representing a structural causal model by measuring
the dependence of the residuals of a linear regression model between two variables
on the variables
"""
import copy
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import networkx as nx


class NonGaussianOrientation:
    """
    Class to Orient the edges of a DAG representing a structural causal model by measuring
    the dependence of the residuals of a linear regression model between two variables
    on the variables
    """

    def __init__(self):
        pass

    def orient_edges(self, graph: nx.DiGraph, data_frame: pd.DataFrame) -> nx.DiGraph:
        """
        Orient the edges of a DAG representing a structural causal model

        Parameters
        ----------
        graph : nx.DiGraph
            Graph to orient
        data_frame : pd.DataFrame
            Data to use for orientation

        Returns
        -------
        nx.DiGraph
            Oriented graph
        """
        graph = copy.deepcopy(graph)
        done = False
        while not done:
            done = True
            for edge in graph.edges():
                if (edge[1], edge[0]) in graph.edges():
                    done = False
                    info_xy = self.estimate_residual_info(
                        data_frame[edge[0]].values, data_frame[edge[1]].values
                    )
                    info_yx = self.estimate_residual_info(
                        data_frame[edge[1]].values, data_frame[edge[0]].values
                    )

                    if info_xy > info_yx:
                        graph.remove_edge(edge[1], edge[0])
                    else:
                        graph.remove_edge(edge[0], edge[1])

                    break

        # Remove cycles

        while True:
            try:
                cycle = nx.find_cycle(graph)
                graph.remove_edge(cycle[0][0], cycle[0][1])
            except:
                break
        return graph

    def estimate_residual_info(self, x: np.array, y: np.array):
        """
        Estimate the mutual information between the residuals of a linear regression
        and the input variable

        Parameters
        ----------
        x : np.array
            Input variable
        y : np.array
            Output variable

        Returns
        -------
        float
            Mutual information between the residuals of a linear regression and the input variable
        """
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        inp = np.hstack((x, np.ones((x.shape[0], 1))))
        theta = np.linalg.pinv(inp) @ y

        pred = inp @ theta
        residuals = y - pred
        return mutual_info_regression(x.reshape(-1, 1), residuals.reshape(-1))
