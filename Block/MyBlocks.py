import tensorflow as tf


class ResBlock(tf.keras.layers):
    def __init__(self, output_filter):
        super(ResBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=output_filter,
                                            kernel_size=(3, 3),
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=output_filter,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, x, input_filter, output_filter):
        res = self.conv1(x)
        res = self.bn1(res)
        res = tf.nn.relu(res)
        res = self.conv12(res)
        res = self.bn2(res)

        if input_filter == output_filter:
            identity = x
        else:
            identity = tf.keras.layers.Conv2D(filters=output_filter, kernel_size=(1, 1),
                                              padding="same")
        output = tf.keras.layers.add([res, identity])
        output = tf.nn.relu(output)
        return output


