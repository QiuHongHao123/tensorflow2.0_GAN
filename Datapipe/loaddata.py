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

    def preprocess(pre_low, pre_full):
        low_min = tf.reduce_min(pre_low)
        low_max = tf.reduce_max(pre_low)
        pre_low = (pre_low - low_min) / (low_max - low_min)
        full_min = tf.reduce_min(pre_full)
        full_max = tf.reduce_max(pre_full)
        pre_full = (pre_full - full_min) / (full_max - full_min)

        return pre_low, pre_full

    print(len(allLowdose), len(allFulldose))
    fulldose_imgs = path2img(allFulldose)
    lowdose_imgs = path2img(allLowdose)
    if not pair:

        full_ds = tf.data.Dataset.from_tensor_slices(fulldose_imgs)
        low_ds = tf.data.Dataset.from_tensor_slices(lowdose_imgs)

        return low_ds, full_ds
    else:
        total_ds = tf.data.Dataset.from_tensor_slices((lowdose_imgs, fulldose_imgs)).map(preprocess)
        return total_ds


'''
total_ds = loaddata()
for i, (l, f) in enumerate(total_ds):
    if i % 1000 == 0:
        plt.figure()
        plt.subplot(331)
        plt.imshow(l, cmap='gray')
        plt.subplot(332)
        plt.imshow(f, cmap='gray')
        plt.show()

'''
train_db=loaddata()
dbiter = train_db.__iter__()
temp = list(dbiter)
l_show, f_show = temp[0]
plt.imshow(l_show, cmap='gray')
plt.show()
plt.imshow(f_show, cmap='gray')
plt.show()