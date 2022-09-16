import tensorflow as tf

class ImageLoader:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
    
    def __call__(self, path):
        file = tf.io.read_file(path)
        img = tf.image.decode_image(file, expand_animations=False)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize(img, (self.width, self.height))
        img = tf.reshape(img, (self.width, self.height, 3))
        return img
