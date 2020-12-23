import pydicom
import tensorflow as tf
import matplotlib.pyplot as plt
from GAN.Models.Generators import Generator_unet
from GAN.Models.PatchDiscriminator import PatchDiscriminator
from Loss.loss_WGAN import G_loss_WGAN, D_loss_WGAN


def trainStep_OnlyL2(low_img, full_img, G, optimizer):
    with tf.GradientTape() as g_tape:
        noise = G(low_img)
        # denoised = low_img - noise
        # # l2 loss
        # g_l = l2_loss = tf.reduce_mean(tf.losses.mean_squared_error(denoised, full_img))
        #残差思想
        g_l = l2_loss = tf.reduce_mean(tf.losses.mean_squared_error(noise, full_img-low_img))
    gradients_of_generator = g_tape.gradient(g_l, G.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables))
    return g_l


def train_withOnlyL2(train_database: tf.data.Dataset, epochs, batchsize, continue_train=False):
    G = Generator_unet()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    train_database = train_database.shuffle(batchsize * 5).batch(batchsize)

    checkpoint = tf.train.Checkpoint(mymodel_G=G)
    if continue_train:
        checkpoint.restore(tf.train.latest_checkpoint('/content/drive/MyDrive/denoise_withoutGAN/model'))
    checkpointManager = tf.train.CheckpointManager(checkpoint,
                                                   directory='/content/drive/MyDrive/denoise_withoutGAN/model',
                                                   max_to_keep=3)
    g_l = 0
    for epoch in range(epochs):
        # 加载一个batch
        for i, (low_img, full_img) in enumerate(train_database):
            g_l = trainStep_OnlyL2(low_img, full_img, G, optimizer)
        print("epoch:", epoch, " g_l:", g_l)
        if epoch % 5 == 0:
            image_bytes = pydicom.dcmread('/content/drive/MyDrive/test.dcm')
            image = image_bytes.pixel_array
            res = G(image)
            plt.imshow(tf.squeeze(res))
            plt.show()
        if epoch % 10 == 0:
            checkpointManager.save(epoch)
            print("save checkpoint" + str(epoch))


def trainStep_withGAN_G(low_img, full_img, G, D, g_optimizer):
    with tf.GradientTape() as g_tape:
        # noise_img = G(low_img)
        # gen_img = low_img - noise_img
        gen_img = G(low_img)
        gen_out = D(gen_img)
        g_loss = G_loss_WGAN(gen_out)
        l1_loss = tf.reduce_mean(tf.abs(full_img - gen_img))
        l2_loss = tf.reduce_mean(tf.losses.mean_squared_error(gen_img, full_img))
        g_loss = g_loss+200*l1_loss
    gradients_of_generator = g_tape.gradient(g_loss, G.trainable_variables)
    g_optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables))
    return g_loss


def trainStep_withGAN_D(low_img, full_img, G, D, d_optimizer):
    with tf.GradientTape() as d_tape:
        # noise_img = G(low_img)
        # gen_img = low_img - noise_img
        gen_img = G(low_img)
        gen_out = D(gen_img)
        full_out = D(full_img)
        d_loss = D_loss_WGAN(D, gen_img, full_img, gen_out, full_out)
    gradients_of_discriminator = d_tape.gradient(d_loss, D.trainable_variables)
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, D.trainable_variables))
    return d_loss


def train_withWGAN(train_database: tf.data.Dataset, epochs, batchsize, continue_train=False):
    G = Generator_unet()
    D = PatchDiscriminator()
    G_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    D_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    train_database = train_database.shuffle(batchsize * 5).batch(batchsize)

    checkpoint = tf.train.Checkpoint(mymodel_G=G, mymodel_D=D)
    if continue_train:
        checkpoint.restore(tf.train.latest_checkpoint('/content/drive/MyDrive/denoise_withGAN/model'))
    checkpointManager = tf.train.CheckpointManager(checkpoint, directory='/content/drive/MyDrive/denoise_withGAN/model',
                                                   max_to_keep=3)

    for epoch in range(epochs):
        g_l = 0
        d_l = 0
        # 加载一个batch
        for i, (low_img, full_img) in enumerate(train_database):
            # train step
            if i % 5 == 0:
                g_l += trainStep_withGAN_G(low_img, full_img, G, D, G_optimizer)
            d_l += trainStep_withGAN_D(low_img, full_img, G, D, D_optimizer)
        print("epoch:", epoch, " g_l:", g_l / i, "d_l", d_l / i)
        if epoch % 5 == 0:
            image_bytes = pydicom.dcmread('/content/drive/MyDrive/test.dcm')
            image = image_bytes.pixel_array
            image = tf.reshape(image, [1, 512, 512, 1])
            image = tf.cast(image, tf.float32)
            res = G(image)
            plt.imshow(tf.squeeze(image - res), cmap='gray')
            plt.show()
            plt.imshow(tf.squeeze(res), cmap='gray')
            plt.show()
        if epoch % 10 == 0:
            checkpointManager.save(epoch)
            print("save checkpoint" + str(epoch))
