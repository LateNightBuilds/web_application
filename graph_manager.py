import json
from typing import List, Tuple, Any, Dict

import networkx as nx
from algorithms.graph.minimum_spanning_tree.methods import MinimumSpanningTreeMethod
from algorithms.graph.minimum_spanning_tree.prim_algorithm import PrimAlgorithm
from algorithms.graph.search.breadth_first_search import BreadthFirstSearch
from algorithms.graph.search.depth_first_search import DepthFirstSearch
from algorithms.graph.search.methods import SearchMethod
from algorithms.graph.shortest_path.a_star import AStar
from algorithms.graph.shortest_path.dijkstra import Dijkstra
from algorithms.graph.shortest_path.methods import ShortestPathMethod
from algorithms.graph.utils.definitions import GridCellType
from algorithms.graph.utils.history import HistoryLogger
from algorithms.graph.utils.utils import graph_to_grid, grid_to_graph
from flask import jsonify


def shortest_path_algorithm(data: Any):
    algorithm_name = data['algorithm']
    grid_data = data['grid']

    has_start = False
    has_end = False
    for cell in grid_data:
        if cell['type'] == 'start':
            has_start = True
        elif cell['type'] == 'end':
            has_end = True

        if has_start and has_end:
            break

    if not (has_start and has_end):
        missing = []
        if not has_start: missing.append("start")
        if not has_end: missing.append("end")
        return jsonify({
            "message": f"Missing required bricks: {', '.join(missing)}"
        }), 400

    graph = Converter.shortest_path_data_to_graph(html_grid=grid_data)
    method = Converter.algorithm_name_to_algorithm_method(algorithm_name=algorithm_name)
    cost, history = run_shortest_path_algorithm(g=graph, method=method)
    formatted_history = FormatHistoryToFrontend.format_shortest_path_algorithm(history=history)

    result_data = {
        'algorithm': algorithm_name,
        'grid': grid_data,
        'history': formatted_history,
        'cost': cost
    }

    # Save the grid data as JSON for future reference
    with open('grid_data.json', 'w') as f:
        json.dump(result_data, f, indent=2)

    return jsonify({
        "message": f"Algorithm {algorithm_name} completed successfully. Path cost: {cost}",
        "history": formatted_history,
        "cost": cost
    })


def graph_search_algorithm(data: Any):
    algorithm_name = data['algorithm']
    graph_data = data['graph']

    has_start = any(node.get('isStart', False) for node in graph_data['nodes'])
    has_end = any(node.get('isEnd', False) for node in graph_data['nodes'])

    if not (has_start and has_end):
        missing = []
        if not has_start: missing.append("start node")
        if not has_end: missing.append("end node")
        return jsonify({
            "message": f"Missing required elements: {', '.join(missing)}"
        }), 400

    graph = Converter.graph_search_data_to_graph(graph_data=graph_data)
    method = Converter.algorithm_name_to_algorithm_method(algorithm_name=algorithm_name)
    start_node = next(node['id'] for node in graph_data['nodes'] if node['isStart'])
    success, history = run_graph_search_algorithm(g=graph, start_node=start_node, method=method)

    formatted_history = FormatHistoryToFrontend.format_graph_search_algorithm(history=history)

    result_data = {
        'algorithm': algorithm_name,
        'grid': graph_data,
        'history': formatted_history,
        'success': success
    }

    # Save the grid data as JSON for future reference
    with open('grid_data.json', 'w') as f:
        json.dump(result_data, f, indent=2)

    return jsonify({
        "message": f"Algorithm {algorithm_name} completed successfully. Path connected: {success}",
        "history": formatted_history,
        "success": success
    })


def minimum_spanning_tree(data: Any):
    algorithm_name = data['algorithm']
    graph_data = data['graph']

    graph = Converter.minimum_spanning_tree_data_to_graph(graph_data=graph_data)
    method = Converter.algorithm_name_to_algorithm_method(algorithm_name=algorithm_name)
    mst_result, history = run_minimum_spanning_tree(g=graph, method=method)
    formatted_history = FormatHistoryToFrontend.format_minimum_spanning_tree(history=history)

    total_weight = sum(weight for _, _, weight in mst_result)

    result_data = {
        'algorithm': algorithm_name,
        'graph': graph_data,
        'history': formatted_history,
        'total_weight': total_weight,
        'mst_edges': mst_result
    }

    # Save the MST data as JSON for future reference
    with open('mst_data.json', 'w') as f:
        json.dump(result_data, f, indent=2)

    return jsonify({
        "message": f"MST Algorithm {algorithm_name} completed successfully.",
        "history": formatted_history,
        "total_weight": total_weight,
        "mst_edges": list(mst_result)
    })


def run_shortest_path_algorithm(g: nx.Graph, method: ShortestPathMethod) -> Tuple[int, HistoryLogger]:
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


def run_graph_search_algorithm(g, start_node, method: SearchMethod) -> tuple[bool, HistoryLogger]:
    graph_search = None
    if method == SearchMethod.BFS:
        graph_search = BreadthFirstSearch(graph=g)
    elif method == SearchMethod.DFS:
        graph_search = DepthFirstSearch(graph=g)

    if not graph_search:
        raise ValueError('Not Implemented')

    return graph_search.run(start_node=start_node)


def run_minimum_spanning_tree(g: nx.Graph, method: MinimumSpanningTreeMethod):
    mst = None
    if method == MinimumSpanningTreeMethod.PRIM:
        mst = PrimAlgorithm(graph=g)

    if not mst:
        raise ValueError('Not Implemented')

    return mst.run()


class FormatHistoryToFrontend:
    @staticmethod
    def format_shortest_path_algorithm(history: HistoryLogger) -> Dict[int, List[int]]:
        formatted_history = {}
        for step, node in history.history_dict.items():
            formatted_history[step] = list(node)

        return formatted_history

    @staticmethod
    def format_graph_search_algorithm(history: HistoryLogger) -> Dict[int, List[int]]:
        formatted_history = {}

        for step, node in history.history_dict.items():
            formatted_history[step] = [node[0], node[1]] if hasattr(node, '__getitem__') else [0, 0]

        return formatted_history

    @staticmethod
    def format_minimum_spanning_tree(history: HistoryLogger) -> Dict[int, List[int]]:
        formatted_history = {}

        for step, entry in history.history_dict.items():
            if isinstance(entry, tuple) and len(entry) >= 2:
                formatted_history[step] = [entry[0], entry[1]]
            else:
                formatted_history[step] = [0, 0]

        return formatted_history


class Converter:

    @staticmethod
    def algorithm_name_to_algorithm_method(algorithm_name: str) -> (ShortestPathMethod
                                                                    | SearchMethod
                                                                    | MinimumSpanningTreeMethod
                                                                    | None):
        algorithm_map = {
            'dijkstra': ShortestPathMethod.DIJKSTRA,
            'a_star': ShortestPathMethod.A_STAR,
            'bfs': SearchMethod.BFS,
            'dfs': SearchMethod.DFS,
            'prims': MinimumSpanningTreeMethod.PRIM
        }
        return algorithm_map.get(algorithm_name)

    @staticmethod
    def shortest_path_data_to_graph(html_grid: List[Dict]) -> nx.Graph:
        processed_grid = [[GridCellType.OPEN_PATH for _ in range(5)] for _ in range(5)]

        html_cell_type_to_grid_cell_type = {'start': GridCellType.START,
                                            'end': GridCellType.END,
                                            'open_path': GridCellType.OPEN_PATH,
                                            'block': GridCellType.BLOCK,
                                            'obstacle': GridCellType.OBSTACLE}

        for cell in html_grid:
            row = cell['row']
            col = cell['col']
            cell_type = cell['type']
            processed_grid[row][col] = html_cell_type_to_grid_cell_type[cell_type]

        return grid_to_graph(input_data=processed_grid)

    @staticmethod
    def graph_search_data_to_graph(graph_data: Dict) -> nx.DiGraph:
        graph = nx.DiGraph()
        [graph.add_edge(u_of_edge=edge['node1'],
                        v_of_edge=edge['node2']) for edge in graph_data['edges']]

        return graph

    @staticmethod
    def minimum_spanning_tree_data_to_graph(graph_data: Dict) -> nx.Graph:
        graph = nx.Graph()
        [graph.add_node(node['id']) for node in graph_data['nodes']]

        [graph.add_edge(u_of_edge=edge['node1'],
                        v_of_edge=edge['node2'],
                        weight=edge.get('weight', 1)) for edge in graph_data['edges']]

        return graph
