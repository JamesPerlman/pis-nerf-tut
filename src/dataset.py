import tensorflow as tf
from src.image_loader import ImageLoader
from src.rays import GetRays

from pathlib import Path
from src.utils import read_json

class Dataset:
    def __init__(self, data_dir: Path | str, near: int, far: int, n_coarse_samples: int):
        self.data_dir = Path(data_dir)
        self.transforms_path = self.data_dir / "transforms.json"
        
        json_data = read_json(self.transforms_path)
        
        self.full_img_paths = [str(self.data_dir / Path(f["file_path"])) for f in json_data["frames"]]
        self.full_c2ws = [f["transform_matrix"] for f in json_data["frames"]]
        
        self.img_width = json_data["w"]
        self.img_height = json_data["h"]
        focal_len = json_data["fl_x"]

        self.get_rays = GetRays(focal_len, self.img_width, self.img_height, near, far, n_coarse_samples)
        self.load_img = ImageLoader(self.img_width, self.img_height)
        
        # split dataset into test, train, and validation
        # 10/10/80 split
        self.test = self.make_split(lambda i: i % 10 == 0)
        self.val = self.make_split(lambda i: i % 10 == 1)
        self.train = self.make_split(lambda i: i % 10 >= 2)

    def make_split(self, predicate):

        c2ws = [self.full_c2ws[i] for i in range(len(self.full_c2ws)) if predicate(i)]
        img_paths = [self.full_img_paths[i] for i in range(len(self.full_img_paths)) if predicate(i)]
        
        rays = (
            tf.data.Dataset
                .from_tensor_slices(c2ws)
                .map(
                    self.get_rays,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
        )

        imgs = (
            tf.data.Dataset
                .from_tensor_slices(img_paths)
                .map(
                    self.load_img,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
        )

        return tf.data.Dataset.zip((rays, imgs))
