
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pywt
from GAN.Models.Generators import Generator_unet_d4
import tensorflow as tf


def preprocess(input):


    min = np.min(input)
    max = np.max(input)
    out = (input - min) / (max - min)

    return out
def test():
    image_bytes_F = pydicom.dcmread('./L19-f-1-080.dcm')
    image_f = image_bytes_F.pixel_array
    image_bytes_L = pydicom.dcmread('./L19-l-1-080.dcm')
    image_l = image_bytes_L.pixel_array
    image_f = preprocess(image_f)
    image_l = preprocess(image_l)
    G = Generator_unet_d4()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    checkpoint = tf.train.Checkpoint(mymodel_G=G, optimizer=optimizer)
    checkpoint.restore('./Models/withoutGAN-d4/ckpt-100')
    plt.imshow(image_f,cmap='gray')
    plt.show()
    plt.imshow(image_l, cmap='gray')
    plt.show()
    plt.imshow(image_l-tf.squeeze(G(tf.reshape(image_l,[1,512,512,1]))), cmap='gray')
    plt.show()



test()
