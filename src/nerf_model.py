import tensorflow as tf

class NeRFModel:
    def __init__(
        self,
        n_dim_xyz: int, # number of dimensions for positional encoding of ray origins
        n_dim_dir: int, # num dims for positional encoding of ray directions
        batch_size: int,
        dense_units: int,
        skip_layer: int,
    ):
        # input layer for ray origins
        ray_o_input = tf.keras.layers.Input(
            shape=(None, None, None, 2 * 3 * n_dim_xyz + 3),
            batch_size=batch_size
        )

        # input layer for ray directions
        ray_d_input = tf.keras.layers.Input(
            shape=(None, None, None, 2 * 3 * n_dim_dir + 3),
            batch_size=batch_size
        )

        # create an input for the MLP
        x = ray_o_input
        for i in range(8):
            # build dense layer
            x = tf.keras.layers.Dense(
                units=dense_units,
                activation="relu",
            )(x)

            # check if we need to make a residual connection
            if i % skip_layer == 0 and i > 0:
                # inject residual connection
                x = tf.keras.layers.concatenate([x, ray_o_input], axis=-1)
        
        # get sigma value
        sigma = tf.keras.layers.Dense(units=1, activation="relu")(x)

        # create feature vector
        feature = tf.keras.layers.Dense(units=dense_units)(x)

        # concat feature vector with direction input and send it through a dense layer
        feature = tf.keras.layers.concatenate([feature, ray_d_input], axis=-1)
        x = tf.keras.layers.Dense(units=dense_units // 2, activation="relu")(feature)

        # get the rgb value
        rgb = tf.keras.layers.Dense(units=3, activation="sigmoid")(x)

        # create the NeRF model
        nerf_model = tf.keras.Model(inputs=[ray_o_input, ray_d_input], outputs=[rgb, sigma])

        return nerf_model