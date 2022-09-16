from typing import Tuple
import numpy as np

class Camera:
    def __init__(self, transform_matrix: np.ndarray, focal_len: float, img_width: int, img_height: int):
        self.transform_matrix = np.array(transform_matrix)
        self.focal_len = float(focal_len)
        self.img_width = int(img_width)
        self.img_height = int(img_height)
    
    def get_rays(self, near: float, far: float, n_samples) -> Tuple[np.ndarray, np.ndarray]:
        img_size = np.array([self.img_width, self.img_height])

        # create a grid of coordinate pairs for the centers of each pixel
        img_coords = np.stack(np.mgrid[:self.img_width, :self.img_height], axis=-1) + [0.5, 0.5]

        # normalize from (-1, -1) to (1, 1)
        normal_coords = 2.0 * (img_coords / [self.img_width, self.img_height]) - 1.0

        # ray xy origs in camera-local coordinates
        local_ray_origs_xy = near * normal_coords *  img_size / self.focal_len

        # homogeneous camera-local ray xyz origins
        ray_zw = np.full((self.img_width, self.img_height, 2), [near, 1])
        homog_local_ray_origs = np.concatenate((local_ray_origs_xy, ray_zw), axis=-1)

        # ray origins in world coordinates (xyz, non-homogeneous)
        world_ray_origs = np.matmul(homog_local_ray_origs, self.transform_matrix)[...,:3]

        # ray directions
        world_cam_loc = self.transform_matrix[:,-1]
        world_ray_dirs = world_ray_origs - world_cam_loc[:3]

        # normalize
        world_ray_dirs = world_ray_dirs / np.linalg.norm(world_ray_dirs, axis=2)[...,np.newaxis]

        # get t_values
        t_values = np.linspace(near, far, n_samples)
        noise_shape = list(world_ray_origs.shape[:-1]) + [n_samples]
        noise = np.random.uniform(size=noise_shape) * (far - near) / n_samples
        t_values = t_values + noise

        return (world_ray_origs, world_ray_dirs, t_values)

