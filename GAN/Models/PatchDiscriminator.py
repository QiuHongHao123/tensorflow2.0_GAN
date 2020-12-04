import tensorflow as tf
import tensorflow.keras as keras

from GAN.Models.InstanceNormalization import InstanceNormalization


class PatchDiscriminator(keras.Model):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.conv1_0 = keras.layers.Conv2D(32, 3, 1, activation=tf.nn.leaky_relu,dilation_rate=1)          # 510
        self.conv1_1 = keras.layers.Conv2D(32, 3, 1, activation=tf.nn.leaky_relu,dilation_rate=2)          # 506
        self.conv1_2 = keras.layers.Conv2D(32, 3, 1, activation=tf.nn.leaky_relu,dilation_rate=3)          # 500

        self.down1 = keras.layers.MaxPool2D(2, 2, padding='VALID')                                         # 250

        self.conv2_0 = keras.layers.Conv2D(64, 3, 1, activation=tf.nn.leaky_relu, dilation_rate=1)         # 248
        self.conv2_1 = keras.layers.Conv2D(64, 3, 1, activation=tf.nn.leaky_relu, dilation_rate=2)         # 244
        self.conv2_2 = keras.layers.Conv2D(64, 3, 1, activation=tf.nn.leaky_relu, dilation_rate=3)         # 238

        self.down2 = keras.layers.MaxPool2D(2, 2, padding='VALID')                                         # 119

        self.conv3_0 = keras.layers.Conv2D(128, 3, 1, activation=tf.nn.leaky_relu, dilation_rate=1)        # 117
        self.conv3_1 = keras.layers.Conv2D(128, 3, 1, activation=tf.nn.leaky_relu, dilation_rate=2)        # 113
        self.conv3_2 = keras.layers.Conv2D(128, 3, 1, activation=tf.nn.leaky_relu, dilation_rate=3)        # 107

        self.down3 =  self.down2 = keras.layers.MaxPool2D(2, 2, padding='VALID')    # 53

        self.conv4_0 = keras.layers.Conv2D(256, 3, 1, activation=tf.nn.leaky_relu, dilation_rate=1)  # 51
        self.conv4_1 = keras.layers.Conv2D(256, 3, 1, activation=tf.nn.leaky_relu, dilation_rate=2)  # 47
        self.conv4_2 = keras.layers.Conv2D(256, 3, 1, activation=tf.nn.leaky_relu, dilation_rate=3)  # 41

        self.last = keras.layers.Conv2D(1, 3, 1)   # 39
        self.flatten = keras.layers.Flatten()
        # self.dense = keras.layers.Dense(100,activation=tf.nn.sigmoid)       # 100


    @tf.function
    def call(self, x):
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.down1(x)

        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.down2(x)

        x = self.conv3_0(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.down3(x)

        x = self.conv4_0(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.last(x)

        out = self.flatten(x)
        #res = self.dense(res)

        out = tf.reduce_mean(out,axis=1)
        return out
class PatchDiscriminator_withNormal(keras.Model):
    def __init__(self):
        super(PatchDiscriminator_withNormal, self).__init__()
        self.conv1_0 = keras.layers.Conv2D(32, 3, 1,dilation_rate=1)          # 510
        self.in1_0 = InstanceNormalization()
        self.conv1_1 = keras.layers.Conv2D(32, 3, 1,dilation_rate=2)          # 506
        self.in1_1 = InstanceNormalization()
        self.conv1_2 = keras.layers.Conv2D(32, 3, 1,dilation_rate=3)          # 500
        self.in1_2 = InstanceNormalization()

        self.down1 = keras.layers.MaxPool2D(2, 2, padding='VALID')                                         # 250

        self.conv2_0 = keras.layers.Conv2D(64, 3, 1, dilation_rate=1)         # 248
        self.in2_0 = InstanceNormalization()
        self.conv2_1 = keras.layers.Conv2D(64, 3, 1, dilation_rate=2)         # 244
        self.in2_1 = InstanceNormalization()
        self.conv2_2 = keras.layers.Conv2D(64, 3, 1, dilation_rate=3)         # 238
        self.in2_2 = InstanceNormalization()

        self.down2 = keras.layers.MaxPool2D(2, 2, padding='VALID')                                         # 119

        self.conv3_0 = keras.layers.Conv2D(128, 3, 1, dilation_rate=1)        # 117
        self.in3_0 = InstanceNormalization()
        self.conv3_1 = keras.layers.Conv2D(128, 3, 1, dilation_rate=2)        # 113
        self.in3_1 = InstanceNormalization()
        self.conv3_2 = keras.layers.Conv2D(128, 3, 1, dilation_rate=3)        # 107
        self.in3_2 = InstanceNormalization()

        self.down3 =  self.down2 = keras.layers.MaxPool2D(2, 2, padding='VALID')    # 53

        self.conv4_0 = keras.layers.Conv2D(256, 3, 1, dilation_rate=1)  # 51
        self.in4_0 = InstanceNormalization()
        self.conv4_1 = keras.layers.Conv2D(256, 3, 1, dilation_rate=2)  # 47
        self.in4_1 = InstanceNormalization()
        self.conv4_2 = keras.layers.Conv2D(256, 3, 1, dilation_rate=3)  # 41
        self.in4_2 = InstanceNormalization()

        self.last = keras.layers.Conv2D(1, 3, 1)   # 39
        self.flatten = keras.layers.Flatten()
        # self.dense = keras.layers.Dense(100,activation=tf.nn.sigmoid)       # 100


    @tf.function
    def call(self, x):
        x = tf.nn.leaky_relu(self.in1_0(self.conv1_0(x)))
        x = tf.nn.leaky_relu(self.in1_1(self.conv1_1(x)))
        x = tf.nn.leaky_relu(self.in1_2(self.conv1_2(x)))
        x = self.down1(x)

        x = tf.nn.leaky_relu(self.in2_0(self.conv2_0(x)))
        x = tf.nn.leaky_relu(self.in2_1(self.conv2_1(x)))
        x = tf.nn.leaky_relu(self.in2_2(self.conv2_2(x)))
        x = self.down2(x)

        x = tf.nn.leaky_relu(self.in3_0(self.conv3_0(x)))
        x = tf.nn.leaky_relu(self.in3_1(self.conv3_1(x)))
        x = tf.nn.leaky_relu(self.in3_2(self.conv3_2(x)))
        x = self.down3(x)

        x = tf.nn.leaky_relu(self.in4_0(self.conv4_0(x)))
        x = tf.nn.leaky_relu(self.in4_1(self.conv4_1(x)))
        x = tf.nn.leaky_relu(self.in4_2(self.conv4_2(x)))
        x = self.last(x)

        out = self.flatten(x)
        #res = self.dense(res)

        out = tf.reduce_mean(out,axis=1)
        return out
test=tf.random.uniform([2,512,512,1])
D = PatchDiscriminator()
res=D(test)
print(res)