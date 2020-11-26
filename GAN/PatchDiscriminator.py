import tensorflow as tf
import tensorflow.keras as keras


class PatchDiscriminator(keras.Model):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, 3, 1, activation=tf.nn.leaky_relu,dilation_rate=1)
        self.conv2 = keras.layers.Conv2D(128, 3, 1, activation=tf.nn.leaky_relu,dilation_rate=2)
        self.conv3 = keras.layers.Conv2D(256, 3, 1, activation=tf.nn.leaky_relu,dilation_rate=3)
        self.conv4 = keras.layers.Conv2D(512, 3, 1, activation=tf.nn.leaky_relu)
        self.last = keras.layers.Conv2D(1, 3, 1)
        self.flatten = keras.layers.Flatten()

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        res = self.last(x)
        res = self.flatten(res)
        return res
