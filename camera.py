from typing import Tuple
import numpy as np

class Camera:
    def __init__(self, transform_matrix: np.ndarray, focal_len: float, img_width: float, img_height):
        self.transform_matrix = transform_matrix
        self.focal_len = focal_len
        self.img_width = img_width
        self.img_height = img_height
    
    def get_rays(self, near: float, far: float) -> Tuple[np.ndarray, np.ndarray]:
        img_size = np.array([self.img_width, self.img_height])

        # create a grid of coordinate pairs for the centers of each pixel
        img_coords = np.stack(np.mgrid[:self.width, :self.height], axis=-1) + [0.5, 0.5]

        # normalize from (-1, -1) to (1, 1)
        normal_coords = 2.0 * (img_coords / [self.width, self.height]) - 1.0

        # ray xy origs in camera-local coordinates
        local_ray_origs_xy = near * normal_coords *  img_size / self.focal_len

        # homogeneous camera-local ray xyz origins
        homog_local_ray_origs = np.concatenate((local_ray_origs_xy, [near, 1]))

        # ray origins in world coordinates (xyz, non-homogeneous)
        world_ray_origs = np.matmul(self.transform_matrix, homog_local_ray_origs)[...,:3]

        # ray directions
        world_cam_loc = self.transform_matrix[:,-1]
        world_ray_dirs = world_ray_origs - world_cam_loc

        # normalize
        world_ray_dirs = world_ray_dirs / np.linalg.norm(world_ray_dirs, axis=2)[...,np.newaxis]

        return (world_ray_origs, world_ray_dirs)

