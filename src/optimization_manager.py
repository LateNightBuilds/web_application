from typing import Dict, Any

from algorithms.optmization.simulated_annealing import SimulatedAnnealingTSP, Point
from flask import jsonify


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
