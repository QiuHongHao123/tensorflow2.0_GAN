import os
import glob
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pydicom


def loaddata(pair=True):
    flag = 1
    imagePath = "../LDCT-and-Projection-data"
    if not os.path.exists(imagePath):
        print(imagePath + "not exist")
        return
    allFulldose = glob.glob(imagePath + "/*/*/*-Full dose images-*/*.dcm")
    allLowdose = glob.glob(imagePath + "/*/*/*-Low dose images-*/*.dcm")

    def path2img(imgPaths):
        images = []
        for imgPath in imgPaths:
            image_bytes = pydicom.dcmread(imgPath)

            image = image_bytes.pixel_array
            images.append(image)
        images = np.array(images)
        return images
    print(len(allLowdose),len(allFulldose))
    fulldose_imgs = path2img(allFulldose)
    lowdose_imgs = path2img(allLowdose)
    if not pair:

        full_ds = tf.data.Dataset.from_tensor_slices(fulldose_imgs)
        low_ds = tf.data.Dataset.from_tensor_slices(lowdose_imgs)

        return low_ds, full_ds
    else:
        total_ds = tf.data.Dataset.from_tensor_slices((lowdose_imgs, fulldose_imgs))
        return total_ds


total_ds = loaddata()
for i, (l, f) in enumerate(total_ds):
    if i % 1000 == 0:
        plt.figure()
        plt.subplot(331)
        plt.imshow(l, cmap='gray')
        plt.subplot(332)
        plt.imshow(f, cmap='gray')
        plt.show()
