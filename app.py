from flask import Flask, render_template, request, jsonify

from graph_manager import shortest_path_algorithm, graph_search_algorithm, minimum_spanning_tree

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


@app.route('/grid')
def grid_page():
    initialize_grid()
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


@app.route('/run_shortest_path_algorithm', methods=['POST'])
def run_shortest_path_algorithm():
    try:
        data = request.get_json()
        json_file = shortest_path_algorithm(data=data)
        return json_file

    except Exception as e:
        print(f"Error in run_algorithm: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/run_search', methods=['POST'])
def handle_search_algorithm():
    try:
        data = request.get_json()
        json_file = graph_search_algorithm(data=data)
        return json_file

    except Exception as e:
        print(f"Error in run_search_algorithm: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/run_mst', methods=['POST'])
def handle_mst_algorithm():
    try:
        data = request.get_json()
        json_file = minimum_spanning_tree(data=data)
        return json_file


    except Exception as e:
        print(f"Error in run_mst_algorithm: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8081)
