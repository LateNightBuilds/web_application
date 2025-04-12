import json
from typing import List, Dict

import networkx as nx
from algorithms.graph.search.methods import SearchMethod
from algorithms.graph.shortest_path.methods import ShortestPathMethod
from algorithms.graph.minimum_spanning_tree.methods import MinimumSpanningTreeMethod
from algorithms.graph.minimum_spanning_tree.prim_algorithm import PrimAlgorithm
from algorithms.graph.utils.definitions import GridCellType
from algorithms.graph.utils.history import HistoryLogger
from algorithms.graph.utils.utils import grid_to_graph
from flask import Flask, render_template, request, jsonify

from graph_client import run_shortest_path, run_search, run_minimum_spanning_tree  # Add run_mst import

app = Flask(__name__)

# Initialize grid for shortest path
grid = []


def initialize_grid():
    global grid
    grid = [['open_path' for _ in range(5)] for _ in range(5)]
    grid[0][0] = 'start'
    grid[4][4] = 'end'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/grid')
def grid_page():
    return render_template('grid.html')


@app.route('/search')
def search_page():
    return render_template('search.html')


@app.route('/minimum_spanning_tree')
def mst_page():
    return render_template('minimum_spanning_tree.html')  # Add route for MST page


@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()
    row, col, brick_type = data['row'], data['col'], data['type']
    grid[row][col] = brick_type
    return jsonify({"message": "Grid updated!"})


@app.route('/reset', methods=['POST'])
def reset():
    initialize_grid()
    return jsonify({"message": "Grid reset!"})


@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    try:
        data = request.get_json()
        algorithm_name = data['algorithm']
        grid_data = data['grid']

        # Verify that start and end are present
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

        input_data = convert_html_cell_type_to_grid_cell_type(html_grid=grid_data)
        method = convert_algorithm_name_to_algorithm_method(algorithm_name=algorithm_name)
        graph = grid_to_graph(input_data=input_data)
        cost, history = run_shortest_path(g=graph, method=method)

        formatted_history = format_history_for_frontend(history)

        # Create the result object to save
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

    except Exception as e:
        print(f"Error in run_algorithm: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/run_search', methods=['POST'])
def handle_search_algorithm():
    try:
        data = request.get_json()
        algorithm_name = data['algorithm']
        graph_data = data['graph']

        # Verify that start and end nodes are present
        has_start = any(node.get('isStart', False) for node in graph_data['nodes'])
        has_end = any(node.get('isEnd', False) for node in graph_data['nodes'])

        if not (has_start and has_end):
            missing = []
            if not has_start: missing.append("start node")
            if not has_end: missing.append("end node")
            return jsonify({
                "message": f"Missing required elements: {', '.join(missing)}"
            }), 400

        # Run the search algorithm
        graph = convert_graph_data_to_graph(graph_data=graph_data)
        start_node = next(node['id'] for node in graph_data['nodes'] if node['isStart'])
        method = convert_algorithm_name_to_algorithm_method(algorithm_name=algorithm_name)
        success, history = run_search(g=graph, start_node=start_node, method=method)

        formatted_history = format_history_for_frontend(history)

        # Create the result object to save
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

    except Exception as e:
        print(f"Error in run_search_algorithm: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/run_mst', methods=['POST'])
def handle_mst_algorithm():
    try:
        data = request.get_json()
        algorithm_name = data['algorithm']
        graph_data = data['graph']

        graph = convert_graph_data_to_graph_with_weights(graph_data=graph_data)

        method = convert_algorithm_name_to_algorithm_method(algorithm_name=algorithm_name)

        mst_result, history = run_minimum_spanning_tree(g=graph, method=method)

        total_weight = sum(weight for _, _, weight in mst_result)

        formatted_history = format_mst_history_for_frontend(history)

        # Create the result object to save
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

    except Exception as e:
        print(f"Error in run_mst_algorithm: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


def format_history_for_frontend(history: HistoryLogger) -> Dict[int, List[int]]:
    """Convert history to a format suitable for the frontend."""
    formatted_history = {}

    for step, node in history.history_dict.items():
        # If history is in the format {step: node}
        if isinstance(node, tuple):
            formatted_history[step] = list(node)
        # If history is in the format {step: (node, data)}
        elif isinstance(node, tuple) and len(node) == 2:
            formatted_history[step] = list(node[0])
        else:
            # Try to handle any other format
            try:
                formatted_history[step] = [node[0], node[1]] if hasattr(node, '__getitem__') else [0, 0]
            except (TypeError, IndexError):
                formatted_history[step] = [0, 0]  # Default fallback

    return formatted_history


def format_mst_history_for_frontend(history: HistoryLogger) -> Dict[int, List[int]]:
    """Convert MST history to a format suitable for the frontend."""
    # This function might need to be adjusted based on the exact history format
    formatted_history = {}

    # You might want to track edge additions, node selections, etc.
    for step, entry in history.history_dict.items():
        if isinstance(entry, tuple) and len(entry) >= 2:
            formatted_history[step] = [entry[0], entry[1]]
        else:
            formatted_history[step] = [0, 0]

    return formatted_history


def convert_html_cell_type_to_grid_cell_type(html_grid: List[Dict]) -> List[List[GridCellType]]:
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

    return processed_grid


def convert_graph_data_to_graph(graph_data: Dict) -> nx.DiGraph:
    graph = nx.DiGraph()

    [graph.add_edge(u_of_edge=edge['node1'],
                    v_of_edge=edge['node2']) for edge in graph_data['edges']]

    return graph


def convert_graph_data_to_graph_with_weights(graph_data: Dict) -> nx.Graph:
    """
    Convert frontend graph data to a networkx graph with edge weights
    Expected input format:
    {
        'nodes': [{'id': node_id, 'x': x, 'y': y}, ...],
        'edges': [{'node1': source, 'node2': target, 'weight': weight}, ...]
    }
    """
    graph = nx.Graph()

    # Add nodes
    [graph.add_node(node['id']) for node in graph_data['nodes']]

    # Add weighted edges
    [graph.add_edge(u_of_edge=edge['node1'],
                    v_of_edge=edge['node2'],
                    weight=edge.get('weight', 1)) for edge in graph_data['edges']]

    return graph


def convert_algorithm_name_to_algorithm_method(algorithm_name: str):
    """Convert algorithm name to corresponding method enum"""
    algorithm_map = {
        # Shortest Path Algorithms
        'dijkstra': ShortestPathMethod.DIJKSTRA,
        'a_star': ShortestPathMethod.A_STAR,

        # Search Algorithms
        'bfs': SearchMethod.BFS,
        'dfs': SearchMethod.DFS,

        # MST Algorithms
        'prims': MinimumSpanningTreeMethod.PRIM
    }

    return algorithm_map.get(algorithm_name)


if __name__ == '__main__':
    initialize_grid()
    app.run(debug=True, port=8081)