from typing import Dict, Any, Callable

from algorithms.optmization.simulated_annealing import SimulatedAnnealingTSP, Point
from algorithms.optmization.gradient_descent import OneDimensionalGradientDescent, OneDimensionalAdaptiveMovementEstimation

from flask import jsonify
import math

def simulated_annealing_tsp(data: Dict[str, Any]):
    try:
        points = [Point(x=point['x'], y=point['y']) for point in data['points']]
        initial_temp = data.get('initial_temp', 1000)
        cooling_rate = data.get('cooling_rate', 0.995)
        solver = SimulatedAnnealingTSP(points=points, initial_temp=initial_temp, cooling_rate=cooling_rate)
        edge_results = solver.run_simulated_annealing_travel_salesman_problem()
        path = [{"x": edge_results[0][0].x, "y": edge_results[0][0].y}]

        for edge in edge_results:
            path.append({"x": edge[1].x, "y": edge[1].y})

        return jsonify({
            "success": True,
            "path": path
        })

    except Exception as e:
        print(f"Error in Simulated Annealing processing: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


def get_function_by_name(function_name: str) -> Callable[[float], float]:
    """Return a function based on its name."""
    function_map = {
        'quadratic': lambda x: x * x,
        'shifted_quadratic': lambda x: (x - 2) * (x - 2),
        'quartic': lambda x: x ** 4 - 4 * x ** 2 + 3,
        'sine': lambda x: math.sin(x) + 0.1 * x * x,  # Add slight quadratic for global minimum
        'rosenbrock': lambda x: x * x + 10 * (x - 1) * (x - 1),
        'rastrigin': lambda x: x * x + 10 * math.cos(2 * math.pi * x) + 10
    }
    return function_map.get(function_name, function_map['quadratic'])


def gradient_descent_optimization(data: Dict[str, Any]):
    """Handle gradient descent optimization requests."""
    try:
        function_type = data.get('function_type', 'quadratic')
        optimizer_type = data.get('optimizer_type', 'gradient_descent')
        starting_point = float(data.get('starting_point', 0.0))
        learning_rate = float(data.get('learning_rate', 0.1))
        iterations = int(data.get('iterations', 50))

        # Adam-specific parameters
        beta1 = float(data.get('beta1', 0.9))
        beta2 = float(data.get('beta2', 0.999))
        epsilon = float(data.get('epsilon', 1e-8))

        # Get the function to optimize
        f_x = get_function_by_name(function_type)

        # Run optimization based on optimizer type
        if optimizer_type == 'gradient_descent':
            optimizer = OneDimensionalGradientDescent(
                num_iterations=iterations,
                step_size=learning_rate
            )
            path = optimizer.run(f_x=f_x, x_init=starting_point)
        elif optimizer_type == 'adam':
            optimizer = OneDimensionalAdaptiveMovementEstimation(
                num_iterations=iterations,
                alpha=learning_rate,
                first_moment_beta=beta1,
                second_moment_beta=beta2,
                epsilon=epsilon
            )
            path = optimizer.run(f_x=f_x, x_init=starting_point)
        else:
            return jsonify({"success": False, "error": "Unknown optimizer type"}), 400

        # Calculate function values for each point in the path
        function_values = [f_x(x) for x in path]

        return jsonify({
            "success": True,
            "path": path,
            "function_values": function_values,
            "function_type": function_type,
            "optimizer_type": optimizer_type,
            "parameters": {
                "starting_point": starting_point,
                "learning_rate": learning_rate,
                "iterations": iterations,
                "beta1": beta1 if optimizer_type == 'adam' else None,
                "beta2": beta2 if optimizer_type == 'adam' else None,
                "epsilon": epsilon if optimizer_type == 'adam' else None
            }
        })

    except Exception as e:
        print(f"Error in gradient descent optimization: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
