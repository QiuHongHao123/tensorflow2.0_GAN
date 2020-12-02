import tensorflow as tf
import tensorflow.keras as keras


class Generator_unet(keras.Model):
    def __init__(self):
        super(Generator_unet, self).__init__()

        self.convd1_0 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME',activation='relu')
        self.convd1_1 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', dilation_rate=2,activation='relu')
        self.convd1_2 = keras.layers.Conv2D(32, 3, padding='SAME', dilation_rate=3,activation='relu')

        self.down1 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd2_0 = keras.layers.Conv2D(64, 3, padding='SAME',activation='relu')
        self.convd2_1 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=2,activation='relu')
        self.convd2_2 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=3,activation='relu')

        self.down2 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd3_0 = keras.layers.Conv2D(128, 3, padding='SAME',activation='relu')
        self.convd3_1 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=2,activation='relu')
        self.convd3_2 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=3,activation='relu')

        self.down3 = keras.layers.MaxPool2D(2, 2, padding='VALID')

        self.convd4_1 = keras.layers.Conv2D(256, 3, padding='SAME',activation='relu')

        self.convu4_1 = keras.layers.Conv2D(256, 3, padding='SAME',activation='relu')

        self.up1 = keras.layers.UpSampling2D(size=2)

        self.convu3_0 = keras.layers.Conv2D(128, 3, padding='SAME',activation='relu')
        self.convu3_1 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=2,activation='relu')
        self.convu3_2 = keras.layers.Conv2D(128, 3, padding='SAME', dilation_rate=3,activation='relu')

        self.up2 = keras.layers.UpSampling2D(size=2)

        self.convu2_0 = keras.layers.Conv2D(64, 3, padding='SAME',activation='relu')
        self.convu2_1 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=2,activation='relu')
        self.convu2_2 = keras.layers.Conv2D(64, 3, padding='SAME', dilation_rate=3,activation='relu')

        self.up3 = keras.layers.UpSampling2D(size=2)

        self.convu1_0 = keras.layers.Conv2D(32, 3, padding='SAME',activation='relu')
        self.convu1_1 = keras.layers.Conv2D(32, 3, padding='SAME', dilation_rate=2,activation='relu')
        self.convu1_2 = keras.layers.Conv2D(32, 3, padding='SAME', dilation_rate=3,activation='relu')

        self.outconv = keras.layers.Conv2D(1, 1, padding='SAME', activation=tf.nn.tanh)

    @tf.function
    def call(self, x):
        # 下采样
        # 第一层
        input1 = self.convd1_0(x)   # 512*512*32
        od1 = self.convd1_1(input1)
        od1 = self.convd1_2(od1)
        od1 = od1+input1            # 512*512*32
        # 第二层
        input2 = self.down1(od1)    # 256*256*32
        input2 = self.convd2_0(input2)  # 256*256*64
        od2 = self.convd2_1(input2)
        od2 = self.convd2_2(od2)
        od2 = od2+input2       # 256*256*64
        # 第三层
        input3 = self.down2(od2)    # 128*128*64
        input3 = self.convd3_0(input3)  # 128*128*128
        od3 = self.convd3_1(input3)
        od3 = self.convd3_2(od3)
        od3 = od3+input3
        # 第四层
        input4 = self.down3(od3)    # 64*64*128
        input4 = self.convd4_1(input4)  # 64*64*256
        # 上采样
        # 第四层
        od4 = self.convu4_1(input4)
        # 第三层
        uinput3 = self.up1(od4)
        uinput3 = self.convu3_0(uinput3)    # 128*128*128
        ou3 = tf.concat([od3,uinput3],axis=3)
        ou3 = self.convu3_1(ou3)
        ou3 = self.convu3_2(ou3)
        ou3 = ou3+uinput3
        #第二层
        uinput2 = self.up2(ou3)
        uinput2 = self.convu2_0(uinput2)
        ou2 = tf.concat([od2,uinput2],axis=3)
        ou2 = self.convu2_1(ou2)
        ou2 = self.convu2_2(ou2)
        ou2 = ou2+uinput2
        # 第一层
        uinput1 = self.up3(ou2)
        uinput1 = self.convu1_0(uinput1)
        ou1 = tf.concat([od1,uinput1],axis=3)
        ou1 = self.convu1_1(ou1)
        ou1 = self.convu1_2(ou1)
        ou1 = ou1+uinput1
        #输出
        out = self.outconv(ou1)
        return out
