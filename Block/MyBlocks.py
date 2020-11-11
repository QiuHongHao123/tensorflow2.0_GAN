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


class DenseLayer(tf.keras.layers):
    def __init__(self, growth_rate, drop_rate):
        super().__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters=4 * growth_rate,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=growth_rate,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding='same')
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)
        self.listLayers = [self.bn1,
                           tf.keras.layers.Activation("relu"),
                           self.conv1,
                           self.bn2,
                           tf.keras.layers.Activation("relu"),
                           self.conv2,
                           self.dropout]

    def call(self, x):
        y = x
        for layer in self.listLayers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x, y], axis=-1)
        return y


class DenseBlock(tf.keras.layers):
    def __init__(self, num_layers, growth_rate, drop_rate=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.listLayers = []
        for _ in range(self.num_layers):
            self.listLayers.append(DenseLayer(growth_rate=self.growth_rate, drop_rate=self.drop_rate))

    def call(self, x):
        for layer in self.listLayers:
            x = layer(x)
        return x
