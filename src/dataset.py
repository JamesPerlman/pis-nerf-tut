import numpy as np
import tensorflow as tf
from src.rays import GetRays

from pathlib import Path
from PIL import Image
from src.utils import read_json

class Dataset:
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.transforms_path = self.data_dir / "transforms.json"
        
        json_data = read_json(self.transforms_path)
        
        img_paths = [str(self.data_dir / Path(f["file_path"])) for f in json_data["frames"]]

        def load_img(bytes):
            path = bytes.decode('utf-8')
            print(f"Loading image {path}")
            return np.array(Image.open(path))
        
        imgs_full = (
            tf.data.Dataset
                .from_tensor_slices(img_paths)
                .map(
                    lambda path: tf.py_function(load_img, [path], [tf.string]),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
        )

        img_width = json_data["w"]
        img_height = json_data["h"]
        focal_len = json_data["fl_x"]

        c2ws = [f["transform_matrix"] for f in json_data["frames"]]
        
        # TODO: set near, far, n_samples in a config file
        get_rays = GetRays(focal_len, img_width, img_height, 2, 6, 64)
        
        rays_full = (
            tf.data.Dataset
                .from_tensor_slices(c2ws)
                .map(
                    get_rays,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
        )

        
        print(rays_full)