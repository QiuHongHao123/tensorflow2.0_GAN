import tensorflow as tf
import tensorflow.keras as keras

"""
2020/11/12
试一下wgan用来进行去噪
考虑复杂度网络结构先采用比较简单的
generator采用编码器和反编码器的结构

"""


class Generator(keras.Model):
    """
    生成器采用三层编码反编码结构
    """

    def __init__(self):
        super(Generator, self).__init__()
        # input512*512
        self.encode1 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.encode2 = keras.layers.Conv2D(128, 3, padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        self.encode3 = keras.layers.Conv2D(256, 3, padding='same')
        self.bn3 = keras.layers.BatchNormalization()

        self.up1 = keras.layers.UpSampling2D(size=2)
        self.conv1 = keras.layers.Conv2D(128, 3, padding='same', activation='relu')
        self.up2 = keras.layers.UpSampling2D(size=2)
        self.conv2 = keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.up3 = keras.layers.UpSampling2D(size=2)
        self.conv3 = keras.layers.Conv2D(1, 3, padding='same', activation='relu')

    @tf.function
    def call(self, x):
        d1 = tf.nn.max_pool2d(tf.nn.relu(self.bn1(self.encode1(x))), 2)
        d2 = tf.nn.max_pool2d(tf.nn.relu(self.bn2(self.encode2(d1))), 2)
        d3 = tf.nn.max_pool2d(tf.nn.relu(self.bn3(self.encode3(d2))), 2)

        u1 = self.conv1(self.up1(d1))
        u2 = self.conv2(self.up2(u1))
        u3 = self.conv3(self.up3(u2))
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
        d1 = tf.nn.max_pool2d(tf.nn.relu(self.encode1(x)), 2)
        d2 = tf.nn.max_pool2d(tf.nn.relu(self.encode2(d1)), 2)
        d3 = tf.nn.max_pool2d(tf.nn.relu(self.encode3(d2)), 2)
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
        fake_img = g(low_img, True)
        fake_output = d(fake_img, training=True)
        g_l = g_loss(fake_output=fake_output)
        l1_loss = tf.reduce_mean(tf.abs(full_img - fake_img))
        # 引入图像的l1正则
        g_l = g_l + (50 * l1_loss)
    gradients_of_generator = g_tape.gradient(g_l, g.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, g.trainable_variables))
    return g_l


def train(train_database: tf.data.Dataset, epoch):
    D = Discriminator()
    G = Generator()
    g_l = d_l = 0
    for i in range(epoch):
        for (low_img, full_img) in train_database:
            if i % 5 == 0:
                g_l = G_train_step(G, D, low_img, full_img)
            d_l = D_train_step(G, D, low_img, full_img)
    print("epoch:", epoch, " g_l:", g_l, " d_l", d_l)
