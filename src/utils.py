from pathlib import Path
from PIL import Image

import json
import tensorflow as tf

def read_json(json_path: Path):
    with open(json_path) as f:
        return json.load(f)

def read_image(img_path: Path):
    img = Image.open(img_path)
    return tf.keras.preprocessing.image.img_to_array(img)
