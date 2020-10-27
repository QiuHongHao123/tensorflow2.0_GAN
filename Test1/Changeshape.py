'''
11.22
维度变换
'''
import tensorflow as tf
'''
改变视图是神经网络中非常常见的操作，可以通过串联多个 Reshape 操作来实现复杂 逻辑，
但是在通过 Reshape 改变视图时，必须始终记住张量的存储顺序，新视图的维度顺序不能与存储顺序相悖，
否则需要通过交换维度操作将存储顺序同步过来。举个例子，对 于 shape 为[4,32,32,3]的图片数据，
通过 Reshape 操作将 shape 调整为[4,1024,3]，此时视图 的维度顺序为𝑏 − 𝑝𝑖𝑥𝑒𝑙 − 𝑐，张量的存储顺序为[𝑏,ℎ, ,𝑐]
'''
def change_shape():
    x=tf.range(96)
    print(x)
    x=tf.reshape(x,[2,4,4,3])
    print(x)
    #可见张量储存顺序始终未改变
def add_D():
    '''
    增加维度
    只能增加长度为1的维度
    通过 tf.expand_dims(x, axis)可在指定的 axis 轴前可以插入一个新的维度：
    tf.expand_dims 的 axis 为正时，表示在当前维度之前插入一个新维度；为负时，表示当前维度之后插入一个新的维度
    :return:

    '''
    x=tf.random.uniform([4,4],maxval=10,dtype=tf.float32)
    print(x)
    x=tf.expand_dims(x,2)
    y=tf.reshape(x,[4,4,1])
    print(x)
    print(y)
    '''
    tf.squeeze(x, axis)函数，axis
    参数为待删除的维度的索引号
    '''
    x1=tf.squeeze(x,2)
    print(x1)
def exchange_D():
    '''
    有时需要直接调整的存储顺序，即交换 维度(Transpose)。通过交换维度，改变了张量的存储顺序，同时也改变了张量的视图。
    注意会改变储存顺序和前面不一样
    :return:
    '''
    x=tf.random.uniform([3,4,4,3])
    print(x)
    '''
    perm指新的维度索引顺序
    '''
    y=tf.transpose(x,perm=[0,3,1,2])
    print(y)
def copy_in_D():
    '''
    在指定维度上复制一份数据
    :return:
    '''
    b=tf.random.uniform([3],maxval=5)
    print(b)
    b=tf.expand_dims(b,axis=0)
    print(b)
    '''
    通过 tf.tile(b, multiples=[2,1])即可在 axis=0 维度复制 1 次，在 axis=1 维度不复制。
    multiples 分别指定了每个维度上 面的复制倍数，对应位置为 1 表明不复制，为 2 表明新长度为原来的长度的 2 倍，即数据 复制一份
    tile本意为瓷砖贴片比较形象
    '''
    b=tf.tile(b,multiples=[2,1])
    print(b)
    a=tf.random.uniform([2,2],maxval=5)
    print(a)
    a=tf.tile(a,multiples=[2,1])
    print(a)
#change_shape()
#add_D()
#exchange_D()
copy_in_D()