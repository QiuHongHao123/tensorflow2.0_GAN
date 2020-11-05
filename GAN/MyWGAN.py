import tensorflow as tf
import tensorflow.keras as keras
from Datapipe import loadmnist
import matplotlib.pyplot as plt


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense1 = keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense3 = keras.layers.Dense(784, activation=tf.nn.relu)

    @tf.function
    def call(self, x, training=True):
        l1 = self.dense1(x)
        l2 = self.dense2(l1)
        l3 = self.dense3(l2)
        l4 = tf.nn.sigmoid(l3)
        l4 = tf.reshape(l4, [-1, 28, 28])
        return l4


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense1 = keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense3 = keras.layers.Dense(1, activation=tf.nn.relu)

    @tf.function
    def call(self, x, training=True):
        x = tf.reshape(x, [-1, 28, 28])
        l1 = self.dense1(x)
        l2 = self.dense2(l1)
        l3 = self.dense3(l2)
        return l3

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


generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, epsilon=1e-10)
discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, epsilon=1e-10)


@tf.function
def G_train_step(images, generator: Generator, discriminator: Discriminator):
    z = tf.random.normal([images.shape[0], 100], mean=0.0, stddev=1.0)
    with tf.GradientTape() as g_tape:
        fake_img = generator(z, True)
        fake_output = discriminator(fake_img, training=True)
        g_l = g_loss(fake_output=fake_output)
    gradients_of_generator = g_tape.gradient(g_l, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return g_l


@tf.function
def D_train_step(images, generator: Generator, discriminator: Discriminator):
    z = tf.random.normal([images.shape[0], 100], mean=0.0, stddev=1.0)
    with tf.GradientTape() as d_tape:
        fake_img = generator(z, True)
        fake_output = discriminator(fake_img, True)
        real_output = discriminator(images, True)
        d_l = d_loss(real_output, fake_output)
    gradients_of_discriminator = d_tape.gradient(d_l, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    discriminator.to_clip()
    return d_l


def train(train_images: tf.data.Dataset, epochs):
    d = Discriminator()
    g = Generator()
    db_batch = train_images.shuffle(32 * 2).batch(batch_size=32)

    for epoch in range(epochs):

        for i, db in enumerate(db_batch):

            d_l = D_train_step(db, g, d)
            if i % 5 == 4:
                g_l = G_train_step(db, g, d)
        if epoch % 2 == 1:
            print('epoch=', epoch, 'G-loss=', g_l, 'D-loss=', d_l)
        if epoch % 100 == 0:
            z = tf.random.normal([1, 100], mean=0.0, stddev=1.0)
            fake_img = g(z, False)
            fake_img = tf.reshape(fake_img, [28, 28])
            plt.imshow(fake_img, cmap='gray')
            plt.show()


train_db, test_db = loadmnist.mnist_dataset()

train(test_db, 400)
