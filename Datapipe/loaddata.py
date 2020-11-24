import os
import glob
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import h5py


def encode2TfRecord():
    flag = 1
    imagePath = "../LDCT-and-Projection-data"
    if not os.path.exists(imagePath):
        print(imagePath + "not exist")
        return
    allFulldose = glob.glob(imagePath + "/*/*/*-Full dose images-*/*.dcm")
    allLowdose = glob.glob(imagePath + "/*/*/*-Low Dose Images-*/*.dcm")

    def path2img(imgPaths):
        images = []
        for imgPath in imgPaths:
            image_bytes = pydicom.dcmread(imgPath)

            image = image_bytes.pixel_array
            images.append(image)

        def preprocess(input):
            min = tf.reduce_min(input)
            max = tf.reduce_max(input)
            output = (input - min) / (max - min)
            return output

        images = preprocess(np.array(images))
        return images

    print(len(allLowdose), len(allFulldose))
    fulldose_imgs = path2img(allFulldose)
    lowdose_imgs = path2img(allLowdose)
    print(fulldose_imgs.shape, lowdose_imgs.shape)

    # tfrecord
    writer = tf.io.TFRecordWriter('trainData')
    for i in range(len(lowdose_imgs)):
        feature = {  # 建立feature字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList((lowdose_imgs[i], fulldose_imgs[i]))),
        }
        # 通过字典创建example
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # 将example序列化并写入字典
        writer.write(example.SerializeToString())
    writer.close()
    print('db finished')
    return None


def decode(filename):
    dataset = tf.data.TFRecordDataset(filename)
    feature = {  # 建立feature字典
        'image': tf.io.FixedLenFeature([], tf.string)
    }

    # 解码
    def _parse_example(input):
        feature_dic = tf.io.parse_single_example(input, feature)
        imgs = tf.reshape(feature_dic['image'], [2, 512, 512])
        return imgs

    dataset = dataset.map(_parse_example)
    return dataset


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
