from flask import Flask, render_template, request, jsonify

from graph_manager import shortest_path_algorithm, graph_search_algorithm, minimum_spanning_tree
from signal_processing_manager import fast_fourier_transform, generate_signal

app = Flask(__name__)

grid = []


def initialize_grid():
    global grid
    grid = [['open_path' for _ in range(5)] for _ in range(5)]
    grid[0][0] = 'start'
    grid[4][4] = 'end'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/shortest_path')
def shortest_path_page():
    initialize_grid()
    return render_template('shortest_path.html')


@app.route('/graph_search')
def graph_search_page():
    return render_template('graph_search.html')


@app.route('/minimum_spanning_tree')
def minimum_spanning_tree_page():
    return render_template('minimum_spanning_tree.html')  # Add route for MST page


@app.route('/fourier')
def fourier_page():
    return render_template('fourier.html')


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


@app.route('/handle_shortest_path_algorithm', methods=['POST'])
def handle_shortest_path_algorithm():
    try:
        data = request.get_json()
        json_file = shortest_path_algorithm(data=data)
        return json_file

    except Exception as e:
        print(f"Error in Shortest Path Finder: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/handle_search_algorithm', methods=['POST'])
def handle_search_algorithm():
    try:
        data = request.get_json()
        json_file = graph_search_algorithm(data=data)
        return json_file

    except Exception as e:
        print(f"Error in Graph Search: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/handle_mst_algorithm', methods=['POST'])
def handle_mst_algorithm():
    try:
        data = request.get_json()
        json_file = minimum_spanning_tree(data=data)
        return json_file


    except Exception as e:
        print(f"Error in Minimum Spanning Tree: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/fourier_transform/generate_signal', methods=['POST'])
def handle_generate_signal():
    try:
        data = request.get_json()
        json_file = generate_signal(data=data)
        return json_file

    except Exception as e:
        print(f"Error While Generating Signal: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/fourier_transform/calculate_fft', methods=['POST'])
def handle_calculate_fft():
    try:
        data = request.get_json()
        json_file = fast_fourier_transform(data=data)
        return json_file

    except Exception as e:
        print(f"Error in Fast Fourier Transform: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8081)
