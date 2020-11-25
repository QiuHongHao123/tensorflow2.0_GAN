import tensorflow as tf
import tensorflow.keras as keras


class Generator_unet(keras.Model):
    def __init__(self):
        super(Generator_unet, self).__init__()

        self.convd1_1 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME',dilation_rate=2)
        self.convd1_2 = keras.layers.Conv2D(64, 3, padding='SAME')

        self.down1 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd2_1 = keras.layers.Conv2D(128, 3, padding='SAME',dilation_rate=2)
        self.convd2_2 = keras.layers.Conv2D(128, 3, padding='SAME')

        self.down2 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd3_1 = keras.layers.Conv2D(256, 3, padding='SAME',dilation_rate=2)
        self.convd3_2 = keras.layers.Conv2D(256, 3, padding='SAME')

        self.down3 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd4_1 = keras.layers.Conv2D(512, 3, padding='SAME')

        self.convu4_1 = keras.layers.Conv2D(512, 3, padding='SAME')

        self.up1 = keras.layers.UpSampling2D(size=2)

        self.convu3_0 = keras.layers.Conv2D(256, 3, padding='SAME')
        self.convu3_1 = keras.layers.Conv2D(256, 3, padding='SAME',dilation_rate=2)
        self.convu3_2 = keras.layers.Conv2D(256, 3, padding='SAME')

        self.up2 = keras.layers.UpSampling2D(size=2)

        self.convu2_0 = keras.layers.Conv2D(128, 3, padding='SAME')
        self.convu2_1 = keras.layers.Conv2D(128, 3, padding='SAME',dilation_rate=2)
        self.convu2_2 = keras.layers.Conv2D(128, 3, padding='SAME')

        self.up3 = keras.layers.UpSampling2D(size=2)

        self.convu1_0 = keras.layers.Conv2D(64, 3, padding='SAME')
        self.convu1_1 = keras.layers.Conv2D(64, 3, padding='SAME',dilation_rate=2)
        self.convu1_2 = keras.layers.Conv2D(64, 3, padding='SAME')

        self.outconv = keras.layers.Conv2D(1, 1, padding='SAME', activation=tf.nn.tanh)

    @tf.function
    def call(self, x):
        od1 = tf.nn.relu(self.convd1_1(x))
        od1 = tf.nn.relu(self.convd1_2(od1))

        od2 = self.down1(od1)
        od2 = tf.nn.relu(self.convd2_1(od2))
        od2 = tf.nn.relu(self.convd2_2(od2))

        od3 = self.down2(od2)
        od3 = tf.nn.relu(self.convd3_1(od3))
        od3 = tf.nn.relu(self.convd3_2(od3))

        od4 = self.down3(od3)
        od4 = tf.nn.relu(self.convd4_1(od4))

        ou4 = tf.nn.relu(self.convu4_1(od4))

        # 上采样
        ou3 = self.covu3_0(self.up1(ou4))

        ou3 = tf.concat([ou3, od3], axis=3)
        ou3 = tf.nn.relu(self.convu3_1(ou3))
        ou3 = tf.nn.relu(self.convu3_1(ou3))

        # 上采样
        ou2 = self.covu2_0(self.up2(ou3))

        ou2 = tf.concat([ou2, od2], axis=3)
        ou2 = tf.nn.relu(self.convu2_1(ou2))
        ou2 = tf.nn.relu(self.convu2_1(ou2))

        # 上采样
        ou1 = self.covu1_0(self.up1(ou2))

        ou1 = tf.concat([ou1, od1], axis=3)
        ou1 = tf.nn.relu(self.convu2_1(ou1))
        ou1 = tf.nn.relu(self.convu2_1(ou1))

        out = self.outconv(ou1)
        return out
