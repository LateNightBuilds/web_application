import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'algorithms')))

from flask import Flask, render_template, request, jsonify

from src.graph_manager import (shortest_path_algorithm,
                               graph_search_algorithm,
                               minimum_spanning_tree)
from src.machine_learning_manager import regularization_impact
from src.optimization_manager import (simulated_annealing_tsp,
                                      gradient_descent_optimization)
from src.signal_processing_manager import (fast_fourier_transform,
                                           generate_signal,
                                           load_sample,
                                           apply_filter,
                                           sound_radar,
                                           image_compression,
                                           apply_kalman_filter)

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
    return render_template('minimum_spanning_tree.html')


@app.route('/fourier')
def fourier_page():
    return render_template('fourier.html')


@app.route('/sound_processing')
def sound_processing_page():
    return render_template('sound_processing.html')


@app.route('/sound_radar')
def sound_radar_page():
    return render_template('sound_radar.html')


@app.route('/image_compression')
def image_compression_page():
    return render_template('image_compression.html')


@app.route('/kalman_filter')
def kalman_filter_page():
    return render_template('kalman_filter.html')


@app.route('/regularization')
def regularization_page():
    return render_template('regularization.html')


@app.route('/simulated_annealing')
def simulated_annealing_page():
    return render_template('simulated_annealing.html')


@app.route('/gradient_descent')
def gradient_descent_page():
    return render_template('gradient_descent.html')


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


@app.route('/sound_processing/load_sample', methods=['POST'])
def handle_load_sample():
    try:
        data = request.get_json()
        json_file = load_sample(data=data)
        return json_file

    except Exception as e:
        print(f"Error loading sound sample: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/sound_processing/apply_filter', methods=['POST'])
def handle_apply_filter():
    try:
        data = request.get_json()
        json_file = apply_filter(data=data)
        return json_file

    except Exception as e:
        print(f"Error applying filter: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/process_image_compression', methods=['POST'])
def handle_image_compression():
    try:
        data = request.get_json()
        json_file = image_compression(data=data)
        return json_file
    except Exception as e:
        print(f"Error in Image Compression: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/process_sound_radar', methods=['POST'])
def handle_sound_radar():
    try:
        data = request.get_json()
        json_file = sound_radar(data=data)
        return json_file

    except Exception as e:
        print(f"Error in Sound Radar processing: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/kalman_filter/apply', methods=['POST'])
def handle_kalman_filter():
    try:
        data = request.get_json()
        json_file = apply_kalman_filter(data=data)
        return json_file

    except Exception as e:
        print(f"Error applying Kalman filter: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/process_regularization', methods=['POST'])
def handle_regularization():
    try:
        data = request.get_json()
        json_file = regularization_impact(data=data)
        return json_file

    except Exception as e:
        print(f"Error in regularization processing: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/solve_tsp', methods=['POST'])
def handle_run_simulated_annealing_tsp():
    try:
        data = request.get_json()
        json_file = simulated_annealing_tsp(data=data)
        return json_file

    except Exception as e:
        print(f"Error in regularization processing: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/gradient_descent/optimize', methods=['POST'])
def handle_gradient_descent():
    try:
        data = request.get_json()
        json_file = gradient_descent_optimization(data=data)
        return json_file

    except Exception as e:
        print(f"Error in gradient descent optimization: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
