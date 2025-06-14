import io
from typing import Tuple

import numpy as np
from PIL import Image
from scipy.io import wavfile
from supabase import create_client, Client

SOUND_SAMPLES_DATABASE_FOLDER = 'sound-samples'
STOCK_PRICES_SAMPLES_DATABASE_FOLDER = 'stock-prices-samples'
IMAGE_SAMPLES_DATABASE_FOLDER = 'image-samples'


class DataManager:
    def __init__(self, url: str, key: str, bucket_name: str):
        self.url = url
        self.key = key

        self.client = None
        self.bucket_name = bucket_name

    def create_client(self) -> Client:
        self.client: Client = create_client(self.url, self.key)
        return self.client

    def download_sound_sample(self, sample_name: str) -> None | Tuple[int, np.ndarray]:
        if not self.client:
            return None

        response = self.client.storage.from_("algorithms-playground").download(f'{SOUND_SAMPLES_DATABASE_FOLDER}/'
                                                                               f'{sample_name}.wav')
        audio_bytes = io.BytesIO(response)
        sample_rate, audio_data = wavfile.read(audio_bytes)
        return sample_rate, audio_data

    def download_image_sample(self, sample_name: str) -> None | Tuple[float, Image.Image]:
        if not self.client:
            return None

        response = self.client.storage.from_("algorithms-playground").download(f'{IMAGE_SAMPLES_DATABASE_FOLDER}/'
                                                                               f'{sample_name}.jpg')
        image_bytes = io.BytesIO(response)
        image = Image.open(image_bytes)
        image_file_size = len(image_bytes.getvalue()) / 1024  # KB
        return image_file_size, image

    def upload_image(self, file: bytes, folder_name: str, file_name: str):
        if not self.client:
            return None

        self.client.storage.from_(self.bucket_name).upload(f"{folder_name}/{file_name}", file=file)

    def get_object_public_url(self, folder_name: str, file_name: str):
        public_url = self.client.storage.from_(self.bucket_name).get_public_url(f"{folder_name}/{file_name}")
        return public_url
