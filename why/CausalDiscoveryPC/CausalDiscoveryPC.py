"""
Module implementing the PC algorithm for causal discovery
"""
from itertools import combinations

import pandas as pd
import networkx as nx

from .IndependenceTests import test_conditional_independence_pearsons

class CausalDiscoveryPC:
    '''
    PC algorithm for causal discovery given a dataset
    '''
    def __init__(self, data_frame: pd.DataFrame):
        self.data_frame = data_frame

        # Adjacency list
        self.adj = []

        self.name_to_id = {}

        # Get a list of the column names
        self.all_variables = self.data_frame.columns.tolist()

        self.features_count = 0

        # Map names to numbers in a dictionary
        for feature in self.all_variables:
            if feature not in self.name_to_id:
                self.name_to_id[feature] = self.features_count
                self.features_count += 1


    def __test_conditional_independence_pearsons_ids(self, x, y, z):
        conditional_combination_strings = []
        for var_id in z:
            conditional_combination_strings.append(self.all_variables[var_id])
        return test_conditional_independence_pearsons(self.data_frame,
                                                      self.all_variables[x],
                                                      self.all_variables[y],
                                                      conditional_combination_strings)

    def __generate_skeleton(self):
        self.adj = []

        # Create a complete graph
        for i in range(0,self.features_count):
            self.adj.append(set(range(0, self.features_count)))
            self.adj[i].discard(i)
            #print(self.adj[i])

        all_variables_ids = list(range(0, self.features_count))

        # Generate all possible 2 variable combinations
        pairs_of_variables = list(combinations(all_variables_ids, 2))

        # Iterate over all length of conditioning sets
        for length in range(0, len(self.name_to_id)):

            # Loop through each ordered pair of adjacent vertices
            for pair_of_variables in pairs_of_variables:

                # Extract the variables from the row
                var1, var2 = pair_of_variables[0], pair_of_variables[1]

                # Only check connected vertices
                if var1 not in self.adj[var2]:
                    continue

                # Extract variables of to be conditioned on
                conditioning_variables = list(self.adj[var1] | self.adj[var2])
                conditioning_variables.remove(var1)
                conditioning_variables.remove(var2)


                if length > len(conditioning_variables):
                    continue

                # Get all possible combinations of the current length
                conditional_combinations = combinations(conditioning_variables, length)

                # Assume they are dependent
                is_dependent = True

                # Iterate over all conditioning sets of specified length
                for conditional_combination in conditional_combinations:
                    conditional_combination = list(conditional_combination)

                    if self.__test_conditional_independence_pearsons_ids(var1,
                                                                         var2,
                                                                         conditional_combination):
                        #print(f'{var1} and {var2} are dependent | {conditional_combination}')
                        pass
                    else:
                        #print(f'{var1} and {var2} are independent | {conditional_combination}')
                        is_dependent = False
                        break

                # In case of independency remove the edge
                if not is_dependent:
                    self.adj[var1].discard(var2)
                    self.adj[var2].discard(var1)


    def __get_all_triple_paths(self):
        triples = []

        for i in range(0, self.features_count):
            for j in self.adj[i]:
                for k in self.adj[j]:
                    if (i != k) and (i not in self.adj[k]):
                        triples.append((i, j, k))
        return triples


    def __get_immoralities(self) -> list:
        triples = self.__get_all_triple_paths()

        immoralities = []

        for triple in triples:
            if self.__test_conditional_independence_pearsons_ids(triple[0], triple[2], [triple[1]]):
                immoralities.append(triple)

        return immoralities


    def __perform_orientation_on_immoralities(self) -> set:
        immoralities = self.__get_immoralities()

        conflicts = set()
        directed_edges = set()

        for triple in immoralities:
            directed_edges.add((triple[0], triple[1]))
            directed_edges.add((triple[2], triple[1]))

        for edge in directed_edges:
            if edge[::-1] in directed_edges:
                conflicts.add(edge)
            else:
                self.adj[edge[1]].discard(edge[0])
        return conflicts

    def __perform_orientation_on_colliders(self, conflicts:set) -> set:
        triples = self.__get_all_triple_paths()

        new_directed_edges = set()

        for triple in triples:
            if triple[0] not in self.adj[triple[1]] and \
                triple[1] in self.adj[triple[2]] and \
                triple[2] in self.adj[triple[1]] :

                new_directed_edges.add((triple[1], triple[2]))

        for edge in new_directed_edges:
            if edge[::-1] in new_directed_edges:
                conflicts.add(edge)
            elif edge not in conflicts:
                self.adj[edge[1]].discard(edge[0])
        return conflicts

    def get_adjacency_list(self) -> dict:
        '''
        Obtain the adjacency list of the graph

        Returns
        -------
        adjacency_list : dict
            A dictionary of the adjacency list of the graph
        '''
        self.__generate_skeleton()

        conflicts = self.__perform_orientation_on_immoralities()

        self.__perform_orientation_on_colliders(conflicts)

        adjacency_list = {}

        # Converting to a dictionary of names
        for i in range(0, self.features_count):

            list_of_names = []

            for item in self.adj[i]:
                list_of_names.append(self.all_variables[item])

            adjacency_list[self.all_variables[i]] = list_of_names

        return adjacency_list

    def get_networkx_graph(self, draw:bool=False) -> nx.DiGraph:
        '''
        Get the networkx graph (nx.DiGraph) of the graph

        Parameters
        ----------
        draw : bool, by default False
            If True, the graph will be drawn

        Returns
        -------
        nx.DiGraph
            The networkx graph of the graph
        '''
        graph = nx.from_dict_of_lists(self.get_adjacency_list(), create_using=nx.DiGraph())
        if draw:
            nx.draw_circular(graph, with_labels=True)
        return graph
