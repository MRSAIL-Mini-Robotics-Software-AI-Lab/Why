"""
Backdoor Adjustment Module to obtain the adjustment set to estimate the ATE between two variables
"""

import networkx as nx
from itertools import chain, combinations


class BackdoorAdjustment:
    """
    Class to preform backdoor adjustment
    """

    def __init__(self, graph):
        """
        Intialize the backdoor adjuster

        Parameters:
        graph (networkx.classes.digraph.DiGraph): The causal graph we want to adjust to it.

        """
        assert nx.is_directed_acyclic_graph(
            graph
        ), "Passed graph must be a directed acyclic graph"
        self.dag = graph
        self.all_variables = [v for v in graph.nodes]

    def _to_set(self, x):
        """
        Convert input to a Set type

        Parameters:
        X (list/str/set/tuple/None): The causal graph we want to adjust to it.

        Return:
        Set: convert the argument passed.

        """
        # if none return empty set
        if x is None:
            return set([])

        # if string, which means a single variable return a set with single element
        if isinstance(x, str):
            return set([x])

        # if list of variables just simply convert it to a set
        if isinstance(x, list):
            return set(x)

        # if tuple of variables just simply convert it to a set
        if isinstance(x, tuple):
            return set(x)

    def _get_powerset(self, myset):
        """
        Generate the powerset of the passed set
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

        Parameters:
        myset (set/list): list of items to generate the powerset

        Returns:
        list[tuples] representing the powerset of the input

        """
        s = list(myset)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def _classify_structure(self, a, b, c):
        """
        Classify  structure as a chain, fork or collider.

        Parameters:
        a (str): first variable
        b (str): second variable
        c (str): third variable

        Returns:
        str representing the type of structure of the three passed consecutive variables

        """
        if self.dag.has_edge(a, b) and self.dag.has_edge(b, c):
            return "chain"

        if self.dag.has_edge(c, b) and self.dag.has_edge(b, a):
            return "chain"

        if self.dag.has_edge(a, b) and self.dag.has_edge(c, b):
            return "collider"

        if self.dag.has_edge(b, a) and self.dag.has_edge(b, c):
            return "fork"
        print("Couldn't classify structure of { }, { }, { }".format(a, b, c))

    def _is_path_d_seperable(self, path, ds=None):
        """
        Checks if a given path can be blocked using the set of variables ds

        Parameters:
        path (list[str]): The path we need to check
        ds (list/str/tuple/None): Set of variables we are going to condition on

        Returns:
        boolean whether the path is blocked by conditioning over ds

        """
        ds = self._to_set(ds)
        if len(path) < 3:
            return False

            # Loop over every 3 consecutive variables in the path
        # and try to find onew of the three main structures
        for t, m, y in zip(path[:-2], path[1:-1], path[2:]):
            structure = self._classify_structure(t, m, y)

            if structure == "chain" or structure == "fork":
                # if theres a chain or fork, and we condition on the variable in the middle,
                # then the path can be blocked by the conditining
                if m in ds:
                    return True

            if structure == "collider":
                # Make sure we are not conditioning on it or it's descendants
                descendants = nx.descendants(self.dag, m)
                if m not in ds and ds.isdisjoint(descendants):
                    return True
                else:
                    return False

        # Path was not blocked by conditioning on ds or by having a collider in the path
        return False

    def _get_all_backdoor_paths(self, T, Y):
        """
        Get all backdoor paths between two variables

        Parameters:
        T (str): Treatment variable
        Y (str): Outcome variable

        Returns:
        list[list[str]] list of backdoor paths from T to Y, a backdoor path
        is a non causal path between T,Y such that the path will remain if we
        remove all arrows comming out of T.

        """
        backdoors = []

        for path in nx.all_simple_paths(self.dag.to_undirected(reciprocal=False), T, Y):
            # By definition a backdoor path is a non causal path between T,Y such that
            # the path will remain if we remove all arrows comming out of T
            if path[1] in self.dag.predecessors(T):
                backdoors.append(path)
        return backdoors

    def is_valid_backdoor_adjustment(self, T, Y, z_set=None):
        """
        Checks if the passed set blocks all backdoor paths.

        Parameters:
        T (str): Treatment variable
        Y (str): Outcome variable
        z_set (list/str/tuple/None): Set of variables we are going to condition on

        Returns:
        boolean whether the passed adjustment set blocks all backdoor paths between
        the treatment variable and the outcome variable.

        """
        backdoor_paths = self._get_all_backdoor_paths(T, Y)
        for path in backdoor_paths:
            # Backdoor path is not blocked when using z_set
            if not self._is_path_d_seperable(path, z_set):
                return False
        # All backdoor paths are blocked by conditioning on z_set
        return True

    def get_all_backdoor_adjustment_set(self, T, Y, max_size: int = None):
        """
        Finds all valid adjustment sets that blocks all backdoor paths
        between treatment and outcome

        Parameters:
        T (str): Treatment variable
        Y (str): Outcome variable

        Returns:
        list[tuples], every tuple contains a valid backdoor adjustment set,
        to get the minimum adjustment set index the first tuple in the output.

        """
        possible_variables_for_adjustment = (
            self._to_set(self.all_variables)
            - self._to_set(T)
            - self._to_set(Y)
            - set(nx.descendants(self.dag, T))
        )
        valid_adjustment_sets = []
        for adj_set in self._get_powerset(possible_variables_for_adjustment):
            if self.is_valid_backdoor_adjustment(T, Y, adj_set):
                valid_adjustment_sets.append(adj_set)
            if max_size is not None and len(valid_adjustment_sets) > max_size:
                break
        return valid_adjustment_sets
