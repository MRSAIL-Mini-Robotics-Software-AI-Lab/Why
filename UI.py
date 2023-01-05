import copy
import matplotlib.pyplot as plt

from IPython.display import clear_output


class UI:
    class Strings:
        """
        Class for storing string constants used in the UI class
        """
        STRING_NOT_DAG_MESSAGE = "Cycle found, The following edges are causing a cycle"
        STRING_PROMPT = "Remove which edges? (write the node names with spaces seperating them, must be multiple of 2)"
        STRING_PROMPT_EXAMPLE = "\n\ti.e. a b c d will remove the edges ab and cd"
        STRING_PROMPT_NOT_PERMITTED = "Not permitted input, try again"

    def __init__(self, adjacencyList: dict[any, list[any]]):
        """
        Initialize the UI class with an adjacency list and create a directed graph from it

        Parameters:
        -----------
        adjacencyList (dict): adjacency list representation of the graph
        """
        self._adjacencyList: dict[any, list[any]] = adjacencyList
        self.digraph = self.createGraph(adjacencyList)

    def display(self, graph):
        """
        Display the graph using a circular layout

        Parameters:
        -----------
        graph (networkx.classes.digraph.DiGraph): the graph to be displayed
        """
        nx.draw_circular(digraph, with_labels=True)
        plt.show()

    def convertToDag(self) -> None:
        """
        Convert the graph to a Directed Acyclic Graph (DAG) by repeatedly 
        prompting the user to remove cycle-causing edges
        """
        # check if DAG
        isDag, arrangementList = self._isDag(self.digraph)

        # while the graph is not DAG
        while not isDag:
            clear_output()
            # display the graph and print cycle-causing edges
            self.display(self.digraph)
            print(UI.Strings.STRING_NOT_DAG_MESSAGE)
            print(arrangementList)

            # prompting the user for the edge to remove
            inputs = input(UI.Strings.STRING_PROMPT +
                           UI.Strings.STRING_PROMPT_EXAMPLE).split()

            while len(inputs) % 2 != 0:
                print(UI.String.STRING_PROMPT_NOT_PERMITTED)
                inputs = input(UI.Strings.STRING_PROMPT +
                               UI.Strings.STRING_PROMPT_EXAMPLE).split()

            # remove chosen edges
            for i in range(0, len(inputs)-1, 2):
                nodeA, nodeB = inputs[i], inputs[i+1]
                self.digraph.remove_edge(nodeA, nodeB)

            # check if the graph is still not a DAG
            isDag, arrangementList = self._isDag(self.digraph)

        # display the graph
        clear_output()
        self.display(self.digraph)

    def createGraph(self, adjacencyList: dict[any, list[any]]) -> nx.DiGraph:
        """
        Create a directed graph from the adjacency list

        Parameters:
        -----------
        adjacencyList (dict): adjacency list representation of the graph

        Returns:
        --------
        networkx.classes.digraph.DiGraph: directed graph created from the adjacency list
        """
        return nx.from_dict_of_lists(adjacencyList,
                                     create_using=nx.DiGraph())

    def setAdjacnecyList(self, adjacencyList: dict[any, list[any]]) -> None:
        """
        Set the adjacency list for the instance

        Parameters:
        -----------
        adjacencyList (dict): adjacency list representation of the graph
        """
        self._adjacencyList = adjacencyList

    def _isDag(self, digraph: nx.DiGraph) -> list[bool, list[any]]:
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
            cycleEdges = nx.find_cycle(digraph)
            return False, cycleEdges
        except:
            return True, []
