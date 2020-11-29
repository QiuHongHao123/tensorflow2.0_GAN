import numpy as np
import pydicom
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from GAN.Generators import Generator_unet

def train_step_OnlyL2(low_img,full_img,G,optimizer):
    with tf.GradientTape() as g_tape:
        noise = G(low_img)
        denoised = low_img - noise
        # l2 loss
        g_l = l2_loss = tf.reduce_mean(tf.losses.mean_squared_error(denoised, full_img))
    gradients_of_generator = g_tape.gradient(g_l, G.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables))
    return g_l
def train_withOnlyL2(train_database:tf.data.Dataset,epochs,batchsize,continue_train=False):
    G = Generator_unet()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    train_database = train_database.shuffle(batchsize * 5).batch(batchsize)

    checkpoint = tf.train.Checkpoint(mymodel_G=G)
    if continue_train:
        checkpoint.restore(tf.train.latest_checkpoint('/content/drive/MyDrive/denoise_withoutGAN/model'))
    checkpointManager = tf.train.CheckpointManager(checkpoint, directory='/content/drive/MyDrive/denoise_withoutGAN/model', max_to_keep=3)
    g_l = 0
    for epoch in range(epochs):
        # 加载一个batch
        for i, (low_img,full_img) in enumerate(train_database):
            g_l = train_step_OnlyL2(low_img,full_img,G,optimizer)
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

