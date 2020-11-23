import tensorflow as tf
import tensorflow.keras as keras


class Generator_unet(keras.Model):
    def __init__(self):
        super(Generator_unet, self).__init__()

        self.convd1_1 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME')
        self.convd1_2 = keras.layers.Conv2D(64, 3, padding='SAME')

        self.down1 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd2_1 = keras.layers.Conv2D(128, 3, padding='SAME')
        self.convd2_2 = keras.layers.Conv2D(128, 3, padding='SAME')

        self.down2 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd3_1 = keras.layers.Conv2D(256, 3, padding='SAME')
        self.convd3_2 = keras.layers.Conv2D(256, 3, padding='SAME')

        self.down3 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd4_1 = keras.layers.Conv2D(512, 3, padding='SAME')
        self.convd4_2 = keras.layers.Conv2D(512, 3, padding='SAME')

        self.down4 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd5_1 = keras.layers.Conv2D(1024, 3, padding='SAME')

        self.convu5_1 = keras.layers.Conv2D(1024, 3, padding='SAME')

        self.up1 = keras.layers.UpSampling2D(size=2)

        self.convu4_0 = keras.layers.Conv2D(512, 3, padding='SAME')
        self.convu4_1 = keras.layers.Conv2D(512, 3, padding='SAME')
        self.convu4_2 = keras.layers.Conv2D(512, 3, padding='SAME')

        self.up2 = keras.layers.UpSampling2D(size=2)

        self.convu3_0 = keras.layers.Conv2D(256, 3, padding='SAME')
        self.convu3_1 = keras.layers.Conv2D(256, 3, padding='SAME')
        self.convu3_2 = keras.layers.Conv2D(256, 3, padding='SAME')

        self.up3 = keras.layers.UpSampling2D(size=2)

        self.convu2_0 = keras.layers.Conv2D(128, 3, padding='SAME')
        self.convu2_1 = keras.layers.Conv2D(128, 3, padding='SAME')
        self.convu2_2 = keras.layers.Conv2D(128, 3, padding='SAME')

        self.up4 = keras.layers.UpSampling2D(size=2)

        self.convu1_0 = keras.layers.Conv2D(64, 3, padding='SAME')
        self.convu1_1 = keras.layers.Conv2D(64, 3, padding='SAME')
        self.convu1_2 = keras.layers.Conv2D(64, 3, padding='SAME')

    def call(self,x):
        od1 = tf.nn.relu(self.convd1_1(x))
        od1 = tf.nn.relu(self.convd1_2(od1))

        od2 = self.down1(od1)
        od2 = tf.nn.relu(self.convd2_1(od2))
        od2 = tf.nn.relu(self.convd2_2(od2))

        od3 = self.down2(od2)
        od3 = tf.nn.relu(self.convd3_1(od3))
        od3 = tf.nn.relu(self.convd3_2(od3))



