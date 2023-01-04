'''
Backdoor Adjustment Module to obtain the adjustment set to estimate the ATE between two variables
'''
from itertools import chain, combinations
import networkx as nx

class BackdoorAdjustment:
    def __init__ (self,graph):
        assert nx.is_directed_acyclic_graph(graph), "Passed graph must be a direceted acyclic graph"
        self.dag = graph
        self.all_variables = [v for v in graph.nodes]

    def _to_set(self,x):

        # if none return empty set
        if x is None:
            return set([])

        #if string, which means a single variable return a set with single element
        if isinstance(x,str):
            return set([x])

        #if list of variables just simply convert it to a set
        if isinstance(x,list):
            return set(x)

        #if tuple of variables just simply convert it to a set
        if isinstance(x,tuple):
            return set(x)

    def _get_powerset(self,myset):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(myset)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def _classify_structure(self, a, b, c):
        """
        Classify  structure as a chain, fork or collider.
        """
        if self.dag.has_edge(a, b) and self.dag.has_edge(b, c):
            return "chain"

        if self.dag.has_edge(c, b) and self.dag.has_edge(b, a):
            return "chain"

        if self.dag.has_edge(a, b) and self.dag.has_edge(c, b):
            return "collider"

        if self.dag.has_edge(b, a) and self.dag.has_edge(b, c):
            return "fork"
        print(f"Couldn't classify structure of {a}, {b}, {c}")

    def _is_path_d_seperable(self,path,ds = None):
        """
        Checks if a given path can be blocked using the set of variables ds
        """
        ds = self._to_set(ds)
        if len(path) < 3:
            return False

        # Loop over every 3 consecutive variables in the path
        # and try to find onew of the three main structures
        for t,m,y in zip(path[:-2],path[1:-1],path[2:]):
            structure = self._classify_structure(t,m,y)

            if structure == "chain" or structure == "fork":
                # if theres a chain or fork, and we condition on the variable in the middle,
                # then the path can be blocked by the conditining
                if m in ds:
                    return True

            if structure == "collider" :
                # Make sure we are not conditioning on it or it's descendants
                descendants = nx.descendants(self.dag,m)
                if m not in ds and ds.isdisjoint(descendants):
                    return True
                else:
                    return False

        #Path was not blocked by conditioning on ds or by having a collider in the path
        return False

    def _get_all_backdoor_paths(self,T,Y):
        """
        Get all backdoor paths between two variables
        """
        backdoors = []

        for path in nx.all_simple_paths(self.dag.to_undirected(reciprocal=False),T,Y):
            # By definition a backdoor path is a non causal path between T,Y such that
            # the path will remain if we remove all arrows comming out of T
            if path[1] in self.dag.predecessors(T):
                backdoors.append(path)
        return backdoors

    def is_valid_backdoor_adjustment(self,T,Y,z_set=None):
        backdoor_paths = self._get_all_backdoor_paths(T,Y)
        for path in backdoor_paths:
            # Backdoor path is not blocked when using z_set
            if not self._is_path_d_seperable(path,z_set):
                return False
        #All backdoor paths are blocked by conditioning on z_set
        return True

    def get_all_backdoor_adjustment_set(self,T,Y):
        possible_variables_for_adjustment = self._to_set(self.all_variables) - self._to_set(T)-self._to_set(Y) - set(nx.descendants(self.dag, T))
        valid_adjustment_sets = []
        for adj_set in self._get_powerset(possible_variables_for_adjustment):
            if self.is_valid_backdoor_adjustment(T,Y,adj_set):
                valid_adjustment_sets.append(adj_set)
        return valid_adjustment_sets
