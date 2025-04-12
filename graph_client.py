from typing import List, Tuple

import networkx as nx
from algorithms.graph.search.breadth_first_search import BreadthFirstSearch
from algorithms.graph.search.depth_first_search import DepthFirstSearch
from algorithms.graph.search.methods import SearchMethod
from algorithms.graph.shortest_path.a_star import AStar
from algorithms.graph.shortest_path.dijkstra import Dijkstra
from algorithms.graph.shortest_path.methods import ShortestPathMethod
from algorithms.graph.utils.definitions import GridCellType
from algorithms.graph.utils.history import HistoryLogger
from algorithms.graph.utils.utils import graph_to_grid, grid_to_graph


def run_minimum_spanning_tree():
    pass


def run_shortest_path(g: nx.Graph, method: ShortestPathMethod) -> Tuple[int, HistoryLogger]:
    maze: List[List[GridCellType]] = graph_to_grid(graph=g)

    path_finder = None
    if method == ShortestPathMethod.DIJKSTRA:
        path_finder = Dijkstra(maze=maze)
    elif method == ShortestPathMethod.A_STAR:
        path_finder = AStar(maze=maze)

    if not path_finder:
        raise ValueError('Not Implemented')

    path_cost, steps = path_finder.run()
    return path_cost, steps


def run_search(g, start_node, method: SearchMethod) -> bool:
    graph_search = None
    if method == SearchMethod.BFS:
        graph_search = BreadthFirstSearch(graph=g)
    elif method == SearchMethod.DFS:
        graph_search = DepthFirstSearch(graph=g)

    if not graph_search:
        raise ValueError('Not Implemented')

    return graph_search.run(start_node=start_node)


if __name__ == '__main__':
    input_data: List[List[GridCellType]] = [
        [GridCellType.START, GridCellType.OBSTACLE, GridCellType.END],
        [GridCellType.OPEN_PATH, GridCellType.OBSTACLE, GridCellType.OPEN_PATH],
        [GridCellType.OPEN_PATH, GridCellType.OBSTACLE, GridCellType.OPEN_PATH],
        [GridCellType.OPEN_PATH, GridCellType.OBSTACLE, GridCellType.OPEN_PATH],
        [GridCellType.OPEN_PATH, GridCellType.OBSTACLE, GridCellType.OPEN_PATH],
        [GridCellType.OPEN_PATH, GridCellType.OBSTACLE, GridCellType.OPEN_PATH],
        [GridCellType.OPEN_PATH, GridCellType.OPEN_PATH, GridCellType.OPEN_PATH]]

    input_graph = grid_to_graph(input_data=input_data)
    a_star_cost, a_star_history = run_shortest_path(g=input_graph,
                                                    method=ShortestPathMethod.A_STAR)

    dijkstra_star_cost, dijkstra_star_history = run_shortest_path(g=input_graph,
                                                                  method=ShortestPathMethod.DIJKSTRA)
