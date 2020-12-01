import tensorflow as tf
import tensorflow.keras as keras


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

        self.conv4_0 = keras.layers.Conv2D(512, 3, 1, activation=tf.nn.leaky_relu, dilation_rate=1)  # 51

        self.last = keras.layers.Conv2D(1, 3, 1,activation=tf.nn.sigmoid)   # 49
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
        x = self.last(x)

        out = self.flatten(x)
        #res = self.dense(res)

        out = tf.reduce_mean(out,axis=1)
        return out
test=tf.zeros([2,512,512,1])
D = PatchDiscriminator()
res=D(test)
print(res)