import tensorflow as tf

class Camera:
    def __init__(self, transform_matrix, focal_len, img_width, img_height):
        self.transform_matrix = transform_matrix
        self.focal_len = float(focal_len)
        self.img_width = int(img_width)
        self.img_height = int(img_height)
    
    def get_rays(self, near: float, far: float, n_samples):
        img_size = tf.constant([self.img_width, self.img_height], dtype=tf.float32)

        # create a grid of coordinate pairs for the centers of each pixel
        img_coords = tf.stack(
            tf.meshgrid(
                tf.range(0, self.img_width, dtype=tf.float32),
                tf.range(0, self.img_height, dtype=tf.float32),
                indexing="ij"
            ),
            axis=-1
        ) + [0.5, 0.5]

        # normalize from (-1, -1) to (1, 1)
        normal_coords = 2.0 * (img_coords / img_size) - 1.0

        # ray xy origs in camera-local coordinates
        local_ray_origs_xy = near * normal_coords *  img_size / self.focal_len

        # homogeneous camera-local ray xyz origins
        ray_nears = tf.fill((self.img_width, self.img_height, 1), tf.constant(near, dtype=tf.float32))
        ray_zw = tf.concat((ray_nears, tf.ones_like(ray_nears, dtype=tf.float32)), axis=-1)
        homog_local_ray_origs = tf.concat((local_ray_origs_xy, ray_zw), axis=-1)

        # ray origins in world coordinates (xyz, non-homogeneous)
        world_ray_origs = tf.matmul(homog_local_ray_origs, self.transform_matrix)[...,:3]

        # ray directions
        world_cam_loc = self.transform_matrix[:,-1]
        world_ray_dirs = world_ray_origs - world_cam_loc[:3]

        # normalize
        world_ray_dirs = world_ray_dirs / tf.norm(world_ray_dirs, axis=2)[...,tf.newaxis]

        # get t_values
        t_values = tf.cast(tf.linspace(near, far, n_samples), dtype=tf.float32)
        noise_shape = list(world_ray_origs.shape[:-1]) + [n_samples]
        noise = tf.random.uniform(shape=noise_shape, dtype=tf.float32) * (far - near) / n_samples
        t_values = t_values + noise

        return (world_ray_origs, world_ray_dirs, t_values)

