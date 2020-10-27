import tensorflow as tf
'''
11.21
有关向量
'''
def about_add():
    '''
    考虑 2 个输出节点的网络层，我们创建长度为 2 的偏置向量𝒃，并累加在每个输出节点
    '''
    z=tf.random.normal([4,2])
    b=tf.zeros([2])
    '''
    注意到这里 shape 为[4,2]的𝒛和 shape 为[2]的𝒃张量可以直接相加，这是为什么呢？让我们 在 Broadcasting 一节为大家揭秘。 
    '''
    print(z+b)
def about_bias():
    '''
    通过高层接口类 Dense()方式创建的网络层，张量 W 和𝒃存储在类的内部，由类自动创 建并管理。可以通过全连接层的 bias 成员变量查看偏置变量𝒃，
    例如创建输入节点数为 4， 输出节点数为 3 的线性层网络，那么它的偏置向量 b 的长度应为 3
    '''
    fc=tf.keras.layers.Dense(3)  # 创建一层 Wx+b，输出节点为 3
    fc.build(input_shape=(2,4))# 通过 build 函数创建 W,b 张量，输入节点为 4
    print(fc.bias)
def constant_3D():
    '''
    三维的张量一个典型应用是表示序列信号，它的格式是 𝑋 = [𝑏,𝑠𝑒𝑞𝑢𝑒𝑛𝑐𝑒 𝑙𝑒𝑛,𝑓𝑒𝑎𝑡𝑢𝑟𝑒 𝑙𝑒𝑛] 其中𝑏表示序列信号的数量，sequence len 表示序列信号在时间维度上的采样点数，feature len 表示每个点的特征长度。
    考虑自然语言处理中句子的表示，如评价句子的是否为正面情绪的情感分类任务网 络，
    为了能够方便字符串被神经网络处理，一般将单词通过嵌入层(Embedding Layer)编码为固定长度的向量，
    比如“a”编码为某个长度 3 的向量，那么 2 个 等长(单词数为 5)的句子序列可以表示为 shape 为[2,5,3]的 3 维张量，
    其中 2 表示句子个 数，5 表示单词数量，3 表示单词向量的长度：
    :return:
    '''
    (x_train,y_train),(x_test,y_test)=tf.keras.datasets.imdb.load_data(num_words=10000)
    print(x_train.shape)

    x_train=tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=80)
    print(x_train.shape)
    embedding = tf.keras.layers.Embedding(10000, 100)
    out=embedding(x_train)
    print(out.shape)
def constant_4D():
    '''
    4 维张量在卷积神经网络中应用的非常广泛，它用于保存特征图(Feature maps)数据， 格式一般定义为 [𝑏,ℎ,w,𝑐]
    其中𝑏表示输入的数量，h/w分布表示特征图的高宽，𝑐表示特征图的通道数，部分深度学习框架也会使用[𝑏,𝑐,ℎ,w]格式的特征图张量，
    例如 PyTorch。图片数据是特征图的一种， 对于含有 RGB 3 个通道的彩色图片，每张图片包含了 h 行 w 列像素点，每个点需要 3 个数值表示 RGB 通道的颜色强度，
    因此一张图片可以表示为[h,w,3]。如图 4.4 所示，最上层 的图片表示原图，它包含了下面 3 个通道的强度信息。
    :return:
    '''
    # 创建 32x32 的彩色图片输入，个数为 4
    x = tf.random.normal([4,32,32,3])
    # 创建卷积神经网络
    layer = tf.keras.layers.Conv2D(16,kernel_size=3)
    out = layer(x)  # 前向计算
    print(out.shape)  # 输出大小
#about_add()
#about_bias()
#constant_3D()
constant_4D()
