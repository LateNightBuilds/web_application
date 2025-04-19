import io
import os
import time
from typing import Any, Dict

import numpy as np
from PIL import Image
from algorithms.signal_processing.fourier_transform import (run_fast_fourier_transform,
                                                            run_generate_signal, SignalType)
from algorithms.signal_processing.image_compression import ImageCompressor, ImageCompressorMethod
from algorithms.signal_processing.kalman_filter import KalmanFilterDataType
from algorithms.signal_processing.sound_processing import SoundFrequencyFilter, FilterType
from algorithms.signal_processing.sound_radar import SoundRadar, Position
from flask import jsonify
from scipy.signal import chirp

from src.data_manager import DataManager

SOUND_SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'sound_samples')
IMAGE_SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'image_samples')

data_base = DataManager(url="https://yjadnucqabkcptxsbqgr.supabase.co",
                        key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlqYWRudWNxYWJrY3B0"
                            "eHNicWdyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDc5MTQwOSwiZXhwIjoyMDYwMzY3NDA5fQ.79"
                            "PnfXgeQw32Zp_q0e0179M-4TOj5H3c6Y0_bcJHQ88",
                        bucket_name="algorithms-playground")
data_base.create_client()


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

    @staticmethod
    def image_compression_name_to_image_compression_method(compression_name: str) -> ImageCompressorMethod:
        compression_name_map = {
            'fft': ImageCompressorMethod.FFT,
            'wavelet': ImageCompressorMethod.WAVELET
        }
        return compression_name_map.get(compression_name)

    @staticmethod
    def data_type_name_to_data_type(data_type_name: str) -> KalmanFilterDataType:
        data_type_map = {
            'audio': KalmanFilterDataType.AUDIO,
            'stock': KalmanFilterDataType.STOCK
        }
        return data_type_map.get(data_type_name)


def generate_signal(data: Dict[str, Any]):
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


def fast_fourier_transform(data: Dict[str, Any]):
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
    sample_rate, waveform = data_base.download_sound_sample(sample_name=sample_name)

    if len(waveform.shape) > 1 and waveform.shape[1] > 1:
        waveform = np.mean(waveform, axis=1)

    if waveform.dtype != np.float32 and waveform.dtype != np.float64:
        waveform = waveform.astype(np.float32) / (2 ** (8 * waveform.itemsize - 1))

    return jsonify({
        'waveform': waveform.tolist(),
        'filtered_waveform': [],
        'frequency': sample_rate
    })


def apply_filter(data: Dict[str, Any]):
    sample_name = data.get('sample_name', 'piano')
    sample_rate, waveform = data_base.download_sound_sample(sample_name=sample_name)

    filter_name = data.get('filter_type', 'lowpass')
    filter_type = Converter.filter_name_to_filter_type(filter_name=filter_name)

    cutoff_frequency = data.get('cutoff_frequency', 1000)
    center_frequency = data.get('center_frequency', 1000)
    bandwidth = data.get('bandwidth', 500)

    obj = SoundFrequencyFilter()
    filtered_waveform = obj.run_apply_filter(waveform=waveform,
                                             sample_rate=sample_rate,
                                             sample_name=sample_name,
                                             filter_type=filter_type,
                                             cutoff_frequency=cutoff_frequency,
                                             center_frequency=center_frequency,
                                             bandwidth=bandwidth)
    return jsonify({
        'waveform': waveform.tolist(),
        'filtered_waveform': filtered_waveform.tolist(),
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


def image_compression(data: Dict[str, Any]):
    sample_name = data.get('image_name', 'lena')
    image: Image = data_base.download_image_sample(sample_name=sample_name)

    if image is None:
        return jsonify({
            "success": False,
            "error": f"Failed to load image sample: {sample_name}"
        }), 500

    compression_name = data.get('compression_method', 'fft')
    compression_method = Converter.image_compression_name_to_image_compression_method(compression_name=compression_name)
    compression_factor = float(data.get('compression_factor', 0.5))

    compressed_image = None
    image_compressor_obj = ImageCompressor(image=image)

    if compression_method == ImageCompressorMethod.FFT:
        compressed_image = image_compressor_obj.run_fft_compression(
            compression_factor=compression_factor)
    elif compression_method == ImageCompressorMethod.WAVELET:
        compressed_image = image_compressor_obj.run_wavelet_compression(
            compression_factor=compression_factor)

    if compressed_image is None:
        return jsonify({
            "success": False,
            "error": "Failed to compress image"
        }), 500

    compressed_img_buffer = io.BytesIO()
    compressed_image = compressed_image.astype(np.uint8)
    compressed_img_buffer.seek(0)

    timestamp = int(time.time())
    compressed_filename = f"{sample_name}_{compression_name}_{int(compression_factor * 100)}_{timestamp}_compressed.jpg"

    results_folder = "compression-results"
    try:
        data_base.upload_image(file=compressed_img_buffer.getvalue(),
                               folder_name=results_folder, file_name=compressed_filename)

    except Exception as e:
        print(f"Error uploading to storage: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Failed to upload compressed images: {str(e)}"
        }), 500

    original_size = len(compressed_img_buffer.getvalue()) / 1024  # KB
    compressed_size = len(compressed_img_buffer.getvalue()) / 1024  # KB
    compression_ratio = 1.0

    try:
        from skimage.metrics import peak_signal_noise_ratio
        original_array = np.array(image)
        psnr = peak_signal_noise_ratio(original_array, compressed_image)
    except Exception:
        psnr = 0

    original_url = data_base.get_object_public_url(folder_name='image-samples', file_name=f'{sample_name}.jpg')
    compressed_url = data_base.get_object_public_url(folder_name=results_folder, file_name=compressed_filename)

    return jsonify({
        "success": True,
        "original_image_url": original_url,
        "compressed_image_url": compressed_url,
        "frequency_domain_url": '',
        "coefficients_image_url": '',
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "psnr": psnr
    })


def apply_kalman_filter(data: Dict[str, Any]):
    pass
