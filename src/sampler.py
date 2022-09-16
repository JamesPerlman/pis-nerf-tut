import tensorflow as tf

class Sampler:
    def __init__(self, batch_size: int, image_width: int, image_height: int):
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

    # PDF = Probability Density Function
    def sample_pdf(self, t_vals_mid, weights, n_fine):
        # add a small value to the weights to prevent it from becoming NaN
        weights += 1e-5

        # normalize the weights to get the PDF
        pdf = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)

        # from PDF to CDF transformation (cumulative distribution function)
        cdf = tf.cumsum(pdf, axis=-1)

        # prepend CDF with zeroes
        cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], axis=-1)

        # get sample points
        u_shape = [self.batch_size, self.image_height, self.image_width, n_fine]
        u = tf.random.uniform(shape=u_shape)

        # get the indices of the points of u when u is inserted into CDF in a sorted manner
        indices = tf.searchsorted(cdf, u, side="right")

        # define boundaries
        below = tf.maximum(0, indices - 1)
        above = tf.minimum(cdf.shape[-1] - 1, indices)
        indices_g = tf.stack([below, above], axis=-1)

        # gather the cdf according to the indices
        cdf_g = tf.gather(cdf, indices_g, axis=-1, batch_dims=len(indices_g.shape) - 2)

        # gather the t_vals according to the indices
        t_vals_mid_g = tf.gather(t_vals_mid, indices_g, axis=-1, batch_dims=len(indices_g.shape) - 2)

        # create the samples by inverting the CDF
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = t_vals_mid_g[..., 0] + t * (t_vals_mid_g[..., 1] - t_vals_mid_g[..., 0])

        # now return the samples!
        return samples