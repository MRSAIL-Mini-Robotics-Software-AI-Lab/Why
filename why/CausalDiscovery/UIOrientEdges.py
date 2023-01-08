"""
Class for orienting edges in a graph manually by the user
"""
import copy
import matplotlib.pyplot as plt
import networkx as nx

from IPython.display import clear_output


class UIOrientEdges:
    """
    Class for orienting edges in a graph manually by the user
    """

    class Strings:
        """
        Class for storing string constants used in the UIOrientEdges class
        """

        STRING_NOT_DAG_MESSAGE = "Cycle found, The following edges are causing a cycle"
        STRING_PROMPT = "Remove which edges? (write the node names with spaces seperating them, must be multiple of 2)"
        STRING_PROMPT_EXAMPLE = "\n\ti.e. a b c d will remove the edges ab and cd\t:"
        STRING_PROMPT_NOT_PERMITTED = "Not permitted input, try again"

    def __init__(self, graph: nx.Graph):
        """
        Initialize the class with an adjacency list and create a directed graph from it

        Parameters:
        -----------
        graph: nx.Graph
            Graph to orient
        """
        self.digraph = copy.deepcopy(graph)
        self._adjacency_list: dict[any, list[any]] = nx.to_dict_of_lists(graph)

    def display(self, digraph):
        """
        Display the graph using a circular layout

        Parameters:
        -----------
        graph (networkx.classes.digraph.DiGraph): the graph to be displayed
        """
        nx.draw_circular(digraph, with_labels=True)
        plt.show()

    def orient_edges(self) -> None:
        """
        Convert the graph to a Directed Acyclic Graph (DAG) by repeatedly
        prompting the user to remove cycle-causing edges
        """
        # check if DAG
        is_dag, arrangement_list = self._is_dag(self.digraph)

        # while the graph is not DAG
        while not is_dag:
            clear_output()
            # display the graph and print cycle-causing edges
            self.display(self.digraph)
            print(UIOrientEdges.Strings.STRING_NOT_DAG_MESSAGE)
            print(arrangement_list)

            # prompting the user for the edge to remove
            inputs = input(
                UIOrientEdges.Strings.STRING_PROMPT
                + UIOrientEdges.Strings.STRING_PROMPT_EXAMPLE
            ).split()

            while len(inputs) % 2 != 0:
                print(UIOrientEdges.Strings.STRING_PROMPT_NOT_PERMITTED)
                inputs = input(
                    UIOrientEdges.Strings.STRING_PROMPT
                    + UIOrientEdges.Strings.STRING_PROMPT_EXAMPLE
                ).split()

            # remove chosen edges
            for i in range(0, len(inputs) - 1, 2):
                node_a, node_b = inputs[i], inputs[i + 1]
                self.digraph.remove_edge(node_a, node_b)

            # check if the graph is still not a DAG
            is_dag, arrangement_list = self._is_dag(self.digraph)

        # display the graph
        clear_output()
        self.display(self.digraph)
        return self.digraph

    def create_graph(self, adjacency_list: dict[any, list[any]]) -> nx.DiGraph:
        """
        Create a directed graph from the adjacency list

        Parameters:
        -----------
        adjacency_list (dict): adjacency list representation of the graph

        Returns:
        --------
        networkx.classes.digraph.DiGraph: directed graph created from the adjacency list
        """
        return nx.from_dict_of_lists(adjacency_list, create_using=nx.DiGraph())

    def set_adjacency_list(self, adjacency_list: dict[any, list[any]]) -> None:
        """
        Set the adjacency list for the instance

        Parameters:
        -----------
        adjacency_list (dict): adjacency list representation of the graph
        """
        self._adjacency_list = adjacency_list

    def _is_dag(self, digraph: nx.DiGraph) -> list[bool, list[any]]:
        """
        Check if the given graph is a Directed Acyclic Graph (DAG) by
        using topological sort to check for cycles

        Parameters:
        -----------
        digraph (networkx.classes.digraph.DiGraph): the graph to be checked

        Returns:
        --------
        list: a tuple with a boolean value indicating if the graph is a DAG
            and a list of edges causing a cycle (if any)
        """
        # perform topological sort, if it fails -> not DAG else it is a DAG
        try:
            cycle_edges = nx.find_cycle(digraph)
            return False, cycle_edges
        except:
            return True, []
