import tensorflow as tf

def encoder_fn(p, L):
    # build list of positional encodings
    gamma = [p]

    # iterate over the number of dimensions in time
    for i in range(L):
        # insert sin and cos of the product of current dimension and position vector
        gamma.append(tf.sin((2.0 ** i) * p))
        gamma.append(tf.cos((2.0 ** i) * p))
    
    # concat positional encodings into a positional vector
    gamma = tf.concat(gamma, axis=-1)

    # return positional encoding vector
    return gamma

