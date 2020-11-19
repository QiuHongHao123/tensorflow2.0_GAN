import os

import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from Datapipe.loaddata import loaddata

"""
2020/11/12
试一下wgan用来进行去噪
考虑复杂度网络结构先采用比较简单的
generator采用编码器和反编码器的结构
采用三层带着BN层的卷积网络采用same padding方式
降采样采用最大池化每次缩小特征图大小一半

上采样为了避免反卷积操作带来的棋盘效应和GAN的梯度消失采用上采样和卷积的方式代替

判别器采用和生成器降采样类似的结构，在最末端使用全连接层进行判断，由于采用的是WGAN所以添加了clip操作并且不带BN层
优化器采用RMSprop

由于是采用的对称图片有监督的训练方式，所以添加了生成图片和图片之间的L1正则作为惩罚
"""


class Generator(keras.Model):
    """
    生成器采用三层编码反编码结构
    """

    def __init__(self):
        super(Generator, self).__init__()
        # input512*512
        self.encode1 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')
        # self.bn1 = keras.layers.BatchNormalization()
        self.encode2 = keras.layers.Conv2D(128, 3, padding='same')
        # self.bn2 = keras.layers.BatchNormalization()
        self.encode3 = keras.layers.Conv2D(256, 3, padding='same')
        # self.bn3 = keras.layers.BatchNormalization()

        self.up1 = keras.layers.UpSampling2D(size=2)
        self.conv1 = keras.layers.Conv2D(128, 3, padding='same', activation='relu')
        self.up2 = keras.layers.UpSampling2D(size=2)
        self.conv2 = keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.up3 = keras.layers.UpSampling2D(size=2)
        self.conv3 = keras.layers.Conv2D(1, 3, padding='same', activation='relu')

    @tf.function
    def call(self, x):
        x = tf.reshape(x, [-1, 512, 512, 1])
        d1 = tf.nn.relu(self.encode1(x))  # 512x512x64
        o1 = tf.nn.max_pool2d(d1, 2, 2, 'VALID')  # 256x256x64
        d2 = tf.nn.relu(self.encode2(o1))  # 256x256x128
        o2 = tf.nn.max_pool2d(d2, 2, 2, 'VALID')  # 128x128x128
        d3 = tf.nn.relu(self.encode3(o2))  # 128x128x256
        o3 = tf.nn.max_pool2d(d3, 2, 2, 'VALID')  # 64x64x256

        u1 = self.conv1(self.up1(o3))  # 128x128x128
        u2 = self.conv2(tf.concat([self.up2(u1), d2], axis=3))  # 256x256x64
        u3 = self.conv3(tf.concat([self.up3(u2), d1], axis=3))  # 512x512x1
        u3 = tf.reshape(u3, (-1, 512, 512))
        return u3


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.encode1 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')

        self.encode2 = keras.layers.Conv2D(128, 3, padding='same')

        self.encode3 = keras.layers.Conv2D(256, 3, padding='same')

        self.flat = keras.layers.Flatten()
        self.fn = keras.layers.Dense(1)

    @tf.function
    def call(self, x):
        x = tf.reshape(x, [-1, 512, 512, 1])
        d1 = tf.nn.max_pool2d(tf.nn.relu(self.encode1(x)), 2, 2, 'VALID')
        d2 = tf.nn.max_pool2d(tf.nn.relu(self.encode2(d1)), 2, 2, 'VALID')
        d3 = tf.nn.max_pool2d(tf.nn.relu(self.encode3(d2)), 2, 2, 'VALID')
        f = self.flat(d3)
        out = self.fn(f)
        return out

    @tf.function
    def to_clip(self):
        for weight in self.trainable_variables:
            weight.assign(tf.clip_by_value(weight, clip_value_min=-0.01, clip_value_max=0.01))


def g_loss(fake_output):
    return -tf.reduce_mean(fake_output)


def d_loss(real_output, fake_output):
    total_loss = -tf.reduce_mean(real_output) + tf.reduce_mean(
        fake_output)  # 用batch 均值逼近期望 然后依据公式 max  所以取反  -E(real)+E(fake)  做min
    return total_loss


generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=3e-4, epsilon=1e-10)
discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=3e-4, epsilon=1e-10)


def D_train_step(g: Generator, d: Discriminator, low_img, full_img):
    with tf.GradientTape() as d_tape:
        fake_img = g(low_img)
        fake_output = d(fake_img)
        real_output = d(full_img)
        d_l = d_loss(real_output, fake_output)
    gradients_of_discriminator = d_tape.gradient(d_l, d.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d.trainable_variables))
    d.to_clip()
    return d_l


def G_train_step(g: Generator, d: Discriminator, low_img, full_img):
    with tf.GradientTape() as g_tape:
        fake_img = g(low_img)
        fake_output = d(fake_img)
        g_l = g_loss(fake_output=fake_output)
        #l1_loss = tf.reduce_mean(tf.abs(full_img - fake_img))
        l2_loss = tf.reduce_mean(tf.losses.mean_squared_error(fake_img,full_img))
        # 引入图像的l1正则
        # g_l = g_l + 50 * l1_loss
        # 引入图像的l2正则
        # g_l =  l2_loss
    gradients_of_generator = g_tape.gradient(g_l, g.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, g.trainable_variables))
    return g_l


def train(train_database: tf.data.Dataset, epochs, batchsize):
    train_database = train_database.shuffle(batchsize * 5).batch(batchsize)
    D = Discriminator()
    G = Generator()
    g_l = d_l = 0
    for epoch in range(epochs):
        for i, (low_img, full_img) in enumerate(train_database):
            if i % 5 == 0:
                g_l = G_train_step(G, D, low_img, full_img)
            d_l = D_train_step(G, D, low_img, full_img)
        print("epoch:", epoch, " g_l:", g_l, " d_l", d_l)
        if epoch % 5 == 0:
            dbiter = train_db.__iter__()
            temp = list(dbiter)
            l_show, f_show = temp[0]
            plt.imshow(tf.squeeze(G(l_show)), cmap='gray')
            plt.show()
            # plt.imshow(f_show, cmap='gray')
            # plt.show()
        if epoch %11 ==10:
            if not os.path.exists('./WGAN-denoise-CT/g'+str(epoch)):
                os.mkdir('./WGAN-denoise-CT/g'+str(epoch))
            G.save("./WGAN-denoise-CT/g"+str(epoch),save_format="tf")
            if not os.path.exists('./WGAN-denoise-CT/d'+str(epoch)):
                os.mkdir('./WGAN-denoise-CT/d'+str(epoch))
            D.save("./WGAN-denoise-CT/d"+str(epoch),save_format="tf")




def Set_GPU_Memory_Growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置 GPU 显存占用为按需分配
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # 异常处理
            print(e)
    else:
        print('No GPU')


# Set_GPU_Memory_Growth()

train_db = loaddata(pair=True)
"""
for i, (l, f) in enumerate(train_db):
    if i % 1000 == 0:
        plt.figure()
        plt.subplot(111)
        plt.imshow(l, cmap='gray')
        plt.subplot(112)
        plt.imshow(f, cmap='gray')
        plt.show()
"""
print("db finished")
train(train_db, 100, batchsize=2)
