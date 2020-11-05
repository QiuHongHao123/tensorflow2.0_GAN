import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
import matplotlib.pyplot as plt


def mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_train.astype('float32')
    train_db = tf.data.Dataset.from_tensor_slices(x_train)
    test_db = tf.data.Dataset.from_tensor_slices(x_test)
    return train_db, test_db
'''
train,test=mnist_dataset()
train=train.batch(batch_size=100)
for i,dbb in enumerate(train):
    if i%100==99:
        print(tf.shape(dbb))
        plt.imshow(tf.reshape(dbb[0],[28,28]),cmap='gray')
        plt.show()

'''


