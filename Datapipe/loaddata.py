import os
import glob

import numpy as np
import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt


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
        return images

    print(len(allLowdose), len(allFulldose))

    def preprocess(low,full):
        low = tf.cast(low, dtype=tf.float32)
        full = tf.cast(full, dtype=tf.float32)
        l_min = tf.reduce_min(low)
        l_max = tf.reduce_max(low)
        low = (low - l_min) / (l_max - l_min)
        f_min = tf.reduce_min(full)
        f_max = tf.reduce_max(full)
        full = (full - f_min) / (f_max - f_min)
        return low,full
    fulldose_imgs = path2img(allFulldose)
    lowdose_imgs = path2img(allLowdose)

    dataset = tf.data.Dataset.from_tensor_slices((lowdose_imgs,fulldose_imgs))
    dataset.map(preprocess)


    # 序列化
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize_example(low_img,full_img):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        low_img=low_img.numpy().tobytes()
        full_img=full_img.numpy().tobytes()
        feature = {
            'low_img': _bytes_feature(low_img),
            'full_img': _bytes_feature(full_img),
        }

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def tf_serialize_example(l,f):

        tf_string = tf.py_function(
            serialize_example,
            (l,f),  # pass these args to the above function.
            tf.string)  # the return type is `tf.string`.
        return tf.reshape(tf_string, ())


    serialized_dataset = dataset.map(tf_serialize_example)

    filename = 'trainData.tfrecord'
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_dataset)

    print('db finished')
    return None


def decode(filename):
    dataset = tf.data.TFRecordDataset(filename)
    feature = {  # 建立feature字典
        'low_img': tf.io.FixedLenFeature([], tf.string),
        'full_img': tf.io.FixedLenFeature([], tf.string)
    }

    # 解码
    def _parse_example(input):
        feature_dic = tf.io.parse_single_example(input, feature)
        return feature_dic

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
encode2TfRecord()

'''
解码代码示例
'''
dataset=decode('trainData.tfrecord')

dataset=dataset.batch(2)
print(dataset)
batchsize=2
for onebatch in dataset:
    full_img = np.zeros([batchsize,262144],dtype='float32')
    for i,f in enumerate(onebatch['full_img']):

        full_img[i]=np.frombuffer(f.numpy(), dtype='float32')
    print(full_img.shape)
    full_img=tf.reshape(full_img,[-1,512,512,1])
    plt.imshow(full_img[0],cmap='gray')
    plt.show()


