
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pywt
from GAN.Models.Generators import Generator_unet_d4
import tensorflow as tf
import cv2

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
    image_l = preprocess(image_l)
    G = Generator_unet_d4()

    checkpoint = tf.train.Checkpoint(mymodel_G=G)
    checkpoint.restore('./Models/withGAN-l1loss-rate200/ckpt-15')
    plt.imshow(image_f,cmap='gray')
    plt.show()
    plt.imshow(image_l, cmap='gray')
    plt.show()
    plt.imshow(image_l-tf.squeeze(G(tf.reshape(image_l,[1,512,512,1]))), cmap='gray')
    plt.show()


def testInreal():
    image = np.fromfile('./realImage/1.2.168.2101.441.100000073.20200313130928.130928_3D.raw', dtype='int16')
    imageNp = np.reshape(image, [400, 400, 400])
    testImage = imageNp[200]
    testImage = preprocess(testImage)


    G = Generator_unet_d4()

    checkpoint = tf.train.Checkpoint(mymodel_G=G)
    checkpoint.restore('./Models/withGAN-simplerD/ckpt-30')

    noise = G(tf.reshape(testImage,[1,400,400,1]))
    plt.imshow(testImage,cmap='gray')
    plt.show()
   
    plt.imshow(testImage-tf.squeeze(noise), cmap='gray')
    plt.show()
    plt.imshow(tf.squeeze(noise),cmap='gray')
    plt.show()

test()
