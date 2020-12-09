
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
    image_bytes_F = pydicom.dcmread('./test_f.dcm')
    image_f = image_bytes_F.pixel_array
    image_bytes_L = pydicom.dcmread('./test_l.dcm')
    image_l = image_bytes_L.pixel_array
    image_l = preprocess((image_l))
    G = Generator_unet_d4()

    checkpoint = tf.train.Checkpoint(mymodel_G=G)
    checkpoint.restore('./Models/withGAN-d4/ckpt-75')
    plt.imshow(image_f,cmap='gray')
    plt.show()
    plt.imshow(image_l, cmap='gray')
    plt.show()
    plt.imshow(image_l-tf.squeeze(G(tf.reshape(image_l,[1,512,512,1]))), cmap='gray')
    plt.show()



test()
