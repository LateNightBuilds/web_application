import base64
import io
from typing import Any

import numpy as np
from algorithms.neural_network.regularization_impact import MLPClassifierForRegularizationImpact
from flask import jsonify
from matplotlib import pyplot as plt


def regularization_impact(data: dict[str, Any]):
    try:
        regularization = float(data['regularization'])
        X = data['data']
        y = data['class']

        model = MLPClassifierForRegularizationImpact(X=X, y=y, regularization=regularization)
        model.fit()
        predictions = model.predict()

        boundary_image: np.ndarray = model.get_decision_boundary()

        metrics = {
            'accuracy': model.get_model_accuracy_score(predictions=predictions),
            'precision': model.get_model_precision_score(predictions=predictions),
            'recall': model.get_model_recall_score(predictions=predictions),
            'f1': model.get_model_f1_score(predictions=predictions)
        }

        buffer = io.BytesIO()
        plt.imsave(buffer, boundary_image, format='png')
        buffer.seek(0)
        serializable_boundary_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return jsonify({
            'message': 'Regularization processing successful',
            'boundary_image': serializable_boundary_image,
            'metrics': metrics
        })


    except Exception as e:
        print(f"Error in regularization processing: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500
