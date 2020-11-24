import tensorflow as tf
import tensorflow.keras as keras


class FullDiscriminator(keras.Model):
    def __init__(self):
        super(FullDiscriminator, self).__init__()
    @tf.function
    def call(self, input):
        input = tf.reshape(input,[-1,512,512,1])

        return input

