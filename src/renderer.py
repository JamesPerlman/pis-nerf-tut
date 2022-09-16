from typing import Tuple
import tensorflow as tf

class Renderer:
    def __init__(self, batch_size: int, image_width: int, image_height: int):
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

    def composite_ray_samples(self, rgb, sigma, t_vals):
        # squeeze last dimension of sigma
        sigma = sigma[..., 0]

        # calculate delta between adjacent t_vals
        delta = t_vals[..., 1:] - t_vals[..., :-1]
        delta_shape = [self.batch_size, self.image_height, self.image_width, 1]
        delta = tf.concat([delta, tf.broadcast_to([1e10], shape=delta_shape)], axis=-1)

        # calculate alpha from sigma and delta values
        alpha = 1.0 - tf.exp(-sigma * delta)

        # calculate exponential term for easier calculations
        exp_term = 1.0 - alpha
        epsilon = 1e-10

        # calculate transmittance and weights of ray points
        transmittance = tf.math.cumprod(exp_term + epsilon, axis=-1, exclusive=True)
        weights = alpha * transmittance

        # build the image and depth map from the points of the rays
        image = tf.reduce_sum(weights[..., None] * rgb, axis=-2)
        depth = tf.reduce_sum(weights * t_vals, axis=-1)

        # return rgb, depth map and weights
        return (image, depth, weights)
        
