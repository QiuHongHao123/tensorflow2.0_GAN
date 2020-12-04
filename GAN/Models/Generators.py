import tensorflow as tf
import tensorflow.keras as keras
from GAN.Models.InstanceNormalization import InstanceNormalization


class Generator_unet(keras.Model):
    def __init__(self):
        super(Generator_unet, self).__init__()

        self.convd1_0 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', activation='relu')
        self.convd1_1 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', dilation_rate=2,
                                            activation='relu')
        self.convd1_2 = keras.layers.Conv2D(32, 3, padding='SAME', dilation_rate=3, activation='relu')

        self.down1 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd2_0 = keras.layers.Conv2D(64, 3, padding='SAME', activation='relu')
        self.convd2_1 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=2, activation='relu')
        self.convd2_2 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=3, activation='relu')

        self.down2 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd3_0 = keras.layers.Conv2D(128, 3, padding='SAME', activation='relu')
        self.convd3_1 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=2, activation='relu')
        self.convd3_2 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=3, activation='relu')

        self.down3 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd4_1 = keras.layers.Conv2D(256, 3, padding='SAME', activation='relu')

        self.convu4_1 = keras.layers.Conv2D(256, 3, padding='SAME', activation='relu')

        self.up1 = keras.layers.UpSampling2D(size=2)

        self.convu3_0 = keras.layers.Conv2D(128, 3, padding='SAME', activation='relu')
        self.convu3_1 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=2, activation='relu')
        self.convu3_2 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=3, activation='relu')

        self.up2 = keras.layers.UpSampling2D(size=2)

        self.convu2_0 = keras.layers.Conv2D(64, 3, padding='SAME', activation='relu')
        self.convu2_1 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=2, activation='relu')
        self.convu2_2 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=3, activation='relu')

        self.up3 = keras.layers.UpSampling2D(size=2)

        self.convu1_0 = keras.layers.Conv2D(32, 3, padding='SAME', activation='relu')
        self.convu1_1 = keras.layers.Conv2D(32, 3, padding='SAME', dilation_rate=2, activation='relu')
        self.convu1_2 = keras.layers.Conv2D(32, 3, padding='SAME', dilation_rate=3, activation='relu')

        self.outconv = keras.layers.Conv2D(1, 1, padding='SAME', activation=tf.nn.tanh)

    @tf.function
    def call(self, x):
        # 下采样
        # 第一层
        input1 = self.convd1_0(x)  # 512*512*32
        od1 = self.convd1_1(input1)
        od1 = self.convd1_2(od1)
        od1 = od1 + input1  # 512*512*32
        # 第二层
        input2 = self.down1(od1)  # 256*256*32
        input2 = self.convd2_0(input2)  # 256*256*64
        od2 = self.convd2_1(input2)
        od2 = self.convd2_2(od2)
        od2 = od2 + input2  # 256*256*64
        # 第三层
        input3 = self.down2(od2)  # 128*128*64
        input3 = self.convd3_0(input3)  # 128*128*128
        od3 = self.convd3_1(input3)
        od3 = self.convd3_2(od3)
        od3 = od3 + input3
        # 第四层
        input4 = self.down3(od3)  # 64*64*128
        input4 = self.convd4_1(input4)  # 64*64*256
        # 上采样
        # 第四层
        od4 = self.convu4_1(input4)
        # 第三层
        uinput3 = self.up1(od4)
        uinput3 = self.convu3_0(uinput3)  # 128*128*128
        ou3 = tf.concat([od3, uinput3], axis=3)
        ou3 = self.convu3_1(ou3)
        ou3 = self.convu3_2(ou3)
        ou3 = ou3 + uinput3
        # 第二层
        uinput2 = self.up2(ou3)
        uinput2 = self.convu2_0(uinput2)
        ou2 = tf.concat([od2, uinput2], axis=3)
        ou2 = self.convu2_1(ou2)
        ou2 = self.convu2_2(ou2)
        ou2 = ou2 + uinput2
        # 第一层
        uinput1 = self.up3(ou2)
        uinput1 = self.convu1_0(uinput1)
        ou1 = tf.concat([od1, uinput1], axis=3)
        ou1 = self.convu1_1(ou1)
        ou1 = self.convu1_2(ou1)
        ou1 = ou1 + uinput1
        # 输出
        out = self.outconv(ou1)
        return out


class Generator_unet_d4(keras.Model):
    def __init__(self):
        super(Generator_unet_d4, self).__init__()

        self.convd1_0 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME')
        self.ind1_0 = InstanceNormalization()
        self.convd1_1 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', dilation_rate=2)
        self.ind1_1 = InstanceNormalization()
        self.convd1_2 = keras.layers.Conv2D(32, 3, padding='SAME', dilation_rate=3)
        self.ind1_2 = InstanceNormalization()

        self.down1 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd2_0 = keras.layers.Conv2D(64, 3, padding='SAME')
        self.ind2_0 = InstanceNormalization()
        self.convd2_1 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=2)
        self.ind2_1 = InstanceNormalization()
        self.convd2_2 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=3)
        self.ind2_2 = InstanceNormalization()

        self.down2 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd3_0 = keras.layers.Conv2D(128, 3, padding='SAME')
        self.ind3_0 = InstanceNormalization()
        self.convd3_1 = keras.layers.Conv2D(128, 3, padding='SAME')
        self.ind3_1 = InstanceNormalization()
        self.convd3_2 = keras.layers.Conv2D(128, 3, padding='SAME')
        self.ind3_2 = InstanceNormalization()

        self.down3 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd4_1 = keras.layers.Conv2D(256, 3, padding='SAME')
        self.ind4_1 = InstanceNormalization()

        self.convu4_1 = keras.layers.Conv2D(256, 3, padding='SAME')
        self.inu4_1 = InstanceNormalization()

        self.up1 = keras.layers.UpSampling2D(size=2)

        self.convu3_0 = keras.layers.Conv2D(128, 3, padding='SAME')
        self.inu3_0 = InstanceNormalization()
        self.convu3_1 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=2)
        self.inu3_1 = InstanceNormalization()
        self.convu3_2 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=3)
        self.inu3_2 = InstanceNormalization()

        self.up2 = keras.layers.UpSampling2D(size=2)

        self.convu2_0 = keras.layers.Conv2D(64, 3, padding='SAME')
        self.inu2_0 = InstanceNormalization()
        self.convu2_1 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=2)
        self.inu2_1 = InstanceNormalization()
        self.convu2_2 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=3)
        self.inu2_2 = InstanceNormalization()

        self.up3 = keras.layers.UpSampling2D(size=2)

        self.convu1_0 = keras.layers.Conv2D(32, 3, padding='SAME')
        self.inu1_0 = InstanceNormalization()
        self.convu1_1 = keras.layers.Conv2D(32, 3, padding='SAME', dilation_rate=2)
        self.inu1_1 = InstanceNormalization()
        self.convu1_2 = keras.layers.Conv2D(32, 3, padding='SAME', dilation_rate=3)
        self.inu1_2 = InstanceNormalization()

        self.outconv = keras.layers.Conv2D(1, 1, padding='SAME', activation=tf.nn.tanh)

    @tf.function
    def call(self, x):
        # 下采样
        # 第一层
        input1 = tf.nn.relu(self.ind1_0(self.convd1_0(x)))  # 512*512*32
        od1 = tf.nn.relu(self.ind1_1(self.convd1_1(input1)))
        od1 = tf.nn.relu(self.ind1_2(self.convd1_2(od1)))
        od1 = od1 + input1  # 512*512*32
        # 第二层
        input2 = self.down1(od1)  # 256*256*32
        input2 = tf.nn.relu(self.ind2_0(self.convd2_0(input2)))  # 256*256*64
        od2 = tf.nn.relu(self.ind2_1(self.convd2_1(input2)))
        od2 = tf.nn.relu(self.ind2_2(self.convd2_2(od2)))
        od2 = od2 + input2  # 256*256*64
        # 第三层
        input3 = self.down2(od2)  # 128*128*64
        input3 = tf.nn.relu(self.ind3_0(self.convd3_0(input3)))  # 128*128*128
        od3 = tf.nn.relu(self.ind3_1(self.convd3_1(input3)))
        od3 = tf.nn.relu(self.ind3_2(self.convd3_2(od3)))
        od3 = od3 + input3
        # 第四层
        input4 = self.down3(od3)  # 64*64*128
        input4 = tf.nn.relu(self.ind4_1(self.convd4_1(input4)))  # 64*64*256
        # 上采样
        # 第四层
        od4 = tf.nn.relu(self.inu4_1(self.convu4_1(input4)))
        # 第三层
        uinput3 = self.up1(od4)
        uinput3 = tf.nn.relu(self.inu3_0(self.convu3_0(uinput3)))  # 128*128*128
        ou3 = tf.concat([od3, uinput3], axis=3)
        ou3 = tf.nn.relu(self.inu3_1(self.convu3_1(ou3)))
        ou3 = tf.nn.relu(self.inu3_2(self.convu3_2(ou3)))
        ou3 = ou3 + uinput3
        # 第二层
        uinput2 = self.up2(ou3)
        uinput2 = tf.nn.relu(self.inu2_0(self.convu2_0(uinput2)))
        ou2 = tf.concat([od2, uinput2], axis=3)
        ou2 = tf.nn.relu(self.inu2_1(self.convu2_1(ou2)))
        ou2 = tf.nn.relu(self.inu2_2(self.convu2_2(ou2)))
        ou2 = ou2 + uinput2
        # 第一层
        uinput1 = self.up3(ou2)
        uinput1 = tf.nn.relu(self.inu1_0(self.convu1_0(uinput1)))
        ou1 = tf.concat([od1, uinput1], axis=3)
        ou1 = tf.nn.relu(self.inu1_1(self.convu1_1(ou1)))
        ou1 = tf.nn.relu(self.inu1_2(self.convu1_2(ou1)))
        ou1 = ou1 + uinput1
        # 输出
        out = self.outconv(ou1)
        return out


class Generator_unet_d5(keras.Model):
    def __init__(self):
        super(Generator_unet_d5, self).__init__()

        self.convd1_0 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME')
        self.ind1_0 = InstanceNormalization()
        self.convd1_1 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', dilation_rate=2)
        self.ind1_1 = InstanceNormalization()
        self.convd1_2 = keras.layers.Conv2D(32, 3, padding='SAME', dilation_rate=3)
        self.ind1_2 = InstanceNormalization()

        self.down1 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd2_0 = keras.layers.Conv2D(64, 3, padding='SAME')
        self.ind2_0 = InstanceNormalization()
        self.convd2_1 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=2)
        self.ind2_1 = InstanceNormalization()
        self.convd2_2 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=3)
        self.ind2_2 = InstanceNormalization()

        self.down2 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd3_0 = keras.layers.Conv2D(128, 3, padding='SAME')
        self.ind3_0 = InstanceNormalization()
        self.convd3_1 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=2)
        self.ind3_1 = InstanceNormalization()
        self.convd3_2 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=3)
        self.ind3_2 = InstanceNormalization()

        self.down3 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd4_0 = keras.layers.Conv2D(256, 3, padding='SAME')
        self.ind4_0 = InstanceNormalization()
        self.convd4_1 = keras.layers.Conv2D(256, 3, padding='SAME', dilation_rate=2)
        self.ind4_1 = InstanceNormalization()
        self.convd4_2 = keras.layers.Conv2D(256, 3, padding='SAME', dilation_rate=3)
        self.ind4_2 = InstanceNormalization()
        self.down4 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd5_0 = keras.layers.Conv2D(512, 3, padding='SAME')
        self.ind5_0 = InstanceNormalization()
        self.convu5_0 = keras.layers.Conv2D(512, 3, padding='SAME')
        self.inu5_0 = InstanceNormalization()

        self.up0 = keras.layers.UpSampling2D(size=2)

        self.convu4_0 = keras.layers.Conv2D(256, 3, padding='SAME')
        self.inu4_0 = InstanceNormalization()
        self.convu4_1 = keras.layers.Conv2D(256, 3, padding='SAME', dilation_rate=2)
        self.inu4_1 = InstanceNormalization()
        self.convu4_2 = keras.layers.Conv2D(256, 3, padding='SAME', dilation_rate=3)
        self.inu4_2 = InstanceNormalization()

        self.up1 = keras.layers.UpSampling2D(size=2)

        self.convu3_0 = keras.layers.Conv2D(128, 3, padding='SAME')
        self.inu3_0 = InstanceNormalization()
        self.convu3_1 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=2)
        self.inu3_1 = InstanceNormalization()
        self.convu3_2 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=3)
        self.inu3_2 = InstanceNormalization()

        self.up2 = keras.layers.UpSampling2D(size=2)

        self.convu2_0 = keras.layers.Conv2D(64, 3, padding='SAME')
        self.inu2_0 = InstanceNormalization()
        self.convu2_1 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=2)
        self.inu2_1 = InstanceNormalization()
        self.convu2_2 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=3)
        self.inu2_2 = InstanceNormalization()

        self.up3 = keras.layers.UpSampling2D(size=2)

        self.convu1_0 = keras.layers.Conv2D(32, 3, padding='SAME')
        self.inu1_0 = InstanceNormalization()
        self.convu1_1 = keras.layers.Conv2D(32, 3, padding='SAME', dilation_rate=2)
        self.inu1_1 = InstanceNormalization()
        self.convu1_2 = keras.layers.Conv2D(32, 3, padding='SAME', dilation_rate=3)
        self.inu1_2 = InstanceNormalization()

        self.outconv = keras.layers.Conv2D(1, 1, padding='SAME', activation=tf.nn.tanh)

    @tf.function
    def call(self, x):
        # 下采样
        # 第一层
        input1 = tf.nn.relu(self.ind1_0(self.convd1_0(x)))  # 512*512*32
        od1 = tf.nn.relu(self.ind1_1(self.convd1_1(input1)))
        od1 = tf.nn.relu(self.ind1_1(self.convd1_2(od1)))

        od1 = od1 + input1  # 512*512*32
        # 第二层
        input2 = self.down1(od1)  # 256*256*32
        input2 = tf.nn.relu(self.ind2_0(self.convd2_0(input2)))  # 256*256*64
        od2 = tf.nn.relu(self.ind2_1(self.convd2_1(input2)))
        od2 = tf.nn.relu(self.ind2_2(self.convd2_2(od2)))

        od2 = od2 + input2  # 256*256*64
        # 第三层
        input3 = self.down2(od2)  # 128*128*64
        input3 = tf.nn.relu(self.ind3_0(self.convd3_0(input3)))  # 128*128*128
        od3 = tf.nn.relu(self.ind3_1(self.convd3_1(input3)))
        od3 = tf.nn.relu(self.ind3_2(self.convd3_2(od3)))

        od3 = od3 + input3
        # 第四层
        input4 = self.down3(od3)  # 64*64*128
        input4 = tf.nn.relu(self.ind4_0(self.convd4_0(input4)))  # 64*64*256
        od4 = tf.nn.relu(self.ind4_1(self.convd4_1(input4)))
        od4 = tf.nn.relu(self.ind4_2(self.convd4_2(od4)))

        od4 = od4 + input4
        # 第五层
        input5 = self.down4(od4)  # 32*32*256
        od5 = tf.nn.relu(self.ind5_0(self.convd5_0(input5)))
        # 上采样
        # 第五层
        ou5 = tf.nn.relu(self.inu5_0(self.convu5_0(od5)))
        # 第四层
        uinput4 = self.up0(ou5)  # 64*64*512
        uinput4 = tf.nn.relu(self.inu4_0(self.convu4_0(uinput4)))  # 64*64*256
        ou4 = tf.concat([od4, uinput4], axis=3)
        ou4 = tf.nn.relu(self.inu4_1(self.convu4_1(ou4)))
        ou4 = tf.nn.relu(self.inu4_2(self.convu4_2(ou4)))

        ou4 = ou4 + uinput4
        # 第三层
        uinput3 = self.up1(ou4)
        uinput3 = tf.nn.relu(self.inu3_0(self.convu3_0(uinput3)))  # 128*128*128
        ou3 = tf.concat([od3, uinput3], axis=3)
        ou3 = tf.nn.relu(self.inu3_1(self.convu3_1(ou3)))
        ou3 = tf.nn.relu(self.inu3_2(self.convu3_2(ou3)))

        ou3 = ou3 + uinput3
        # 第二层
        uinput2 = self.up2(ou3)
        uinput2 = tf.nn.relu(self.inu2_0(self.convu2_0(uinput2)))
        ou2 = tf.concat([od2, uinput2], axis=3)
        ou2 = tf.nn.relu(self.inu2_1(self.convu2_1(ou2)))
        ou2 = tf.nn.relu(self.inu2_2(self.convu2_2(ou2)))

        ou2 = ou2 + uinput2
        # 第一层
        uinput1 = self.up3(ou2)
        uinput1 = tf.nn.relu(self.inu1_0(self.convu1_0(uinput1)))
        ou1 = tf.concat([od1, uinput1], axis=3)
        ou1 = tf.nn.relu(self.inu1_1(self.convu1_1(ou1)))
        ou1 = tf.nn.relu(self.inu1_2(self.convu1_2(ou1)))

        ou1 = ou1 + uinput1
        # 输出
        out = self.outconv(ou1)
        return out
