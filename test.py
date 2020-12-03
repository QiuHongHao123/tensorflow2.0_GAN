
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pywt


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
    image_f = preprocess(image_f)
    image_l = preprocess(image_l)
    plt.imshow(image_f,cmap='gray')
    plt.show()
    plt.imshow(image_l, cmap='gray')
    plt.show()
    plt.imshow(preprocess(image_l-image_f), cmap='gray')
    plt.show()
    plt.imshow(image_l-(image_l - image_f), cmap='gray')
    plt.show()


test()
