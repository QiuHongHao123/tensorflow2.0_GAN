'''
11.22有关索引和切片

'''
import tensorflow as tf
def index_test():

    x=tf.random.normal([4,32,32,3])
    '''
    当张量的维度数较高时，使用[𝑖][𝑗]...[𝑘]的方式书写不方便，可以采用[𝑖,𝑗,…,𝑘]的方 式索引，它们是等价的。 
    '''
    print(x[0][30][30][1])
    print(x[0,30,30,1])

'''
切片和python3中一模一样
'''
index_test()