import numpy as np
import pydicom
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from GAN.Generators import Generator_unet


def train_withOnlyL2(train_database:tf.data.Dataset,epochs,batchsize,continue_train=False):
    G = Generator_unet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    train_database = train_database.shuffle(batchsize * 5).batch(batchsize)

    checkpoint = tf.train.Checkpoint(mymodel_G=G)
    if continue_train:
        checkpoint.restore(tf.train.latest_checkpoint('./WGAN-denoise-CT/model'))
    checkpointManager = tf.train.CheckpointManager(checkpoint, directory='./WGAN-denoise-CT/model', max_to_keep=3)
    g_l=0
    for epoch in range(epochs):
        # 加载一个batch
        for i, onedb in enumerate(train_database):
            full_img = np.zeros([batchsize, 262144], dtype='float32')
            for j, f in enumerate(onedb['full_img']):
                full_img[j] = np.frombuffer(f.numpy(), dtype='float32')
            full_img = tf.reshape(full_img, [-1, 512, 512, 1])
            low_img = np.zeros([batchsize, 262144], dtype='float32')
            for j, f in enumerate(onedb['low_img']):
                low_img[j] = np.frombuffer(f.numpy(), dtype='float32')
            low_img = tf.reshape(low_img, [-1, 512, 512, 1])

            with tf.GradientTape() as g_tape:
                noise = G(low_img)
                denoised = low_img - noise
                # l2 loss
                g_l = l2_loss = tf.reduce_mean(tf.losses.mean_squared_error(denoised,full_img))
            gradients_of_generator = g_tape.gradient(g_l, G.trainable_variables)
            optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables))
        print("epoch:", epoch, " g_l:", g_l)
        if epoch%5==0:
            image_bytes = pydicom.dcmread('/content/drive/MyDrive/test.dcm')
            image = image_bytes.pixel_array
            res = G(image)
            plt.imshow(tf.squeeze(res))
            plt.show()
        if epoch % 10 == 0:
            checkpointManager.save(epoch)
            print("save checkpoint" + str(epoch))

