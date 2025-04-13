from typing import Any

from algorithms.signal_processing.fourier_transform import (run_fast_fourier_transform,
                                                            run_generate_signal, SignalType)
from flask import jsonify


class Converter:

    @staticmethod
    def signal_name_to_signal_type(signal_name: str) -> SignalType:
        signal_map = {
            'sine': SignalType.SINE,
            'square': SignalType.SQUARE,
            'sawtooth': SignalType.SAWTOOTH,
            'triangle': SignalType.TRIANGLE
        }
        return signal_map.get(signal_name)


def generate_signal(data: Any):
    signal_name = data.get('signal_type', 'sine')
    signal_type = Converter.signal_name_to_signal_type(signal_name=signal_name)

    frequency = float(data.get('frequency', 1.0))
    amplitude = float(data.get('amplitude', 1.0))
    duration = float(data.get('duration', 1.0))
    sampling_rate = float(data.get('sampling_rate', 100.0))
    noise_level = float(data.get('noise_level', 0.0))

    t, y = run_generate_signal(signal_type=signal_type,
                               frequency=frequency,
                               amplitude=amplitude,
                               duration=duration,
                               sampling_rate=sampling_rate,
                               noise_level=noise_level)

    return jsonify({
        'time': t,
        'signal': y
    })


def fast_fourier_transform(data: Any):
    signal = data.get('signal', [])
    sampling_rate = float(data.get('sampling_rate', 100.0))

    if not signal:
        return jsonify({'error': 'No signal provided'}), 400

    freq, magnitude, phase = run_fast_fourier_transform(signal=signal,
                                                        sampling_rate=sampling_rate)

    return jsonify({
        'frequency': freq,
        'magnitude': magnitude,
        'phase': phase
    })
