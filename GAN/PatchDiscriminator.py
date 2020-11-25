import tensorflow as tf
import tensorflow.keras as keras


class PatchDiscriminator(keras.Model):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, 4, 2, activation=tf.nn.leaky_relu)
        self.conv2 = keras.layers.Conv2D(128, 4, 2, activation=tf.nn.leaky_relu)
        self.conv3 = keras.layers.Conv2D(256, 4, 2, activation=tf.nn.leaky_relu)
        self.conv4 = keras.layers.Conv2D(512, 4, 2, activation=tf.nn.leaky_relu)
        self.last = keras.layers.Conv2D(1, 4, 1, activation=tf.nn.sigmoid)

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        res = self.last(x)
        return res
