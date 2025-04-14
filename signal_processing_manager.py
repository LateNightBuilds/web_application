import os
from typing import Any, Dict

import numpy as np
import scipy.io.wavfile as wavfile
from algorithms.signal_processing.fourier_transform import (run_fast_fourier_transform,
                                                            run_generate_signal, SignalType)
from algorithms.signal_processing.sound_processing import SoundFrequencyFilter, FilterType
from algorithms.signal_processing.sound_radar import SoundRadar, Position
from flask import jsonify
from scipy.signal import chirp

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'sound_samples')


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

    @staticmethod
    def filter_name_to_filter_type(filter_name: str) -> FilterType:
        filter_map = {
            'lowpass': FilterType.LOW_PASS,
            'highpass': FilterType.HIGH_PASS,
            'bandpass': FilterType.BAND_PASS,
            'bandstop': FilterType.BAND_STOP
        }
        return filter_map.get(filter_name)


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


def load_sample(data: Dict[str, Any]):
    sample_name = data.get('sample_name', 'piano')

    os.makedirs(SAMPLES_DIR, exist_ok=True)

    sample_path = os.path.join(SAMPLES_DIR, f"{sample_name}.wav")
    sample_rate, waveform = wavfile.read(sample_path)

    if len(waveform.shape) > 1 and waveform.shape[1] > 1:
        waveform = np.mean(waveform, axis=1)

    if waveform.dtype != np.float32 and waveform.dtype != np.float64:
        waveform = waveform.astype(np.float32) / (2 ** (8 * waveform.itemsize - 1))

    return jsonify({
        'waveform': waveform.tolist(),
        'frequency': sample_rate
    })


def apply_filter(data: Dict[str, Any]):
    sample_name = data.get('sample_name', 'piano')
    filter_name = data.get('filter_type', 'lowpass')
    filter_type = Converter.filter_name_to_filter_type(filter_name=filter_name)

    cutoff_frequency = data.get('cutoff_frequency', 1000)
    center_frequency = data.get('center_frequency', 1000)
    bandwidth = data.get('bandwidth', 500)

    obj = SoundFrequencyFilter()
    waveform = obj.run_apply_filter(sample_name=sample_name,
                                    filter_type=filter_type,
                                    cutoff_frequency=cutoff_frequency,
                                    center_frequency=center_frequency,
                                    bandwidth=bandwidth)
    return jsonify({
        'waveform': waveform.tolist(),
        'frequency': 44100
    })


def sound_radar(data: Dict[str, Any]):
    mic1_pos = Position(x=data['mic1_x'], y=data['mic1_y'])
    mic2_pos = Position(x=data['mic2_x'], y=data['mic2_y'])
    mic3_pos = Position(x=data['mic3_x'], y=data['mic3_y'])
    mic4_pos = Position(x=data['mic4_x'], y=data['mic4_y'])

    source_resolution = 44100
    duration = 0.03
    t = np.linspace(0, duration, int(source_resolution * duration), endpoint=False)
    source_signal = chirp(t, f0=500, f1=5000, t1=duration, method='linear')

    sound_radar_obj = SoundRadar(noise_signal=source_signal, noise_resolution=source_resolution,
                                 microphone_positions=[mic1_pos, mic2_pos, mic3_pos, mic4_pos])

    source_position = Position(x=data['source_x'], y=data['source_y'])
    estimated_source_position = sound_radar_obj.run_sound_radar(noise_position=source_position)

    return jsonify({
        "success": True,
        "tdoa_values": {
            "tdoa12": 0,
            "tdoa13": 0,
            "tdoa14": 0,
            "tdoa23": 0
        },
        "estimated_position": {
            "x": estimated_source_position.x,
            "y": estimated_source_position.y
        },
        "error_meters": source_position.euclidean_distance(other_position=estimated_source_position)
    })
