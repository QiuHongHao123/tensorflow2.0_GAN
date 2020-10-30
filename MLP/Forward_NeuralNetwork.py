import numpy as np
import matplotlib.pyplot as plt





def layer_size(X, Y):
    """
    :param X: input dataset of shape (input size, number of examples)  (输入数据集大小（几个属性，样本量）)
    :param Y: labels of shape (output size, number of exmaples) (标签数据大小（标签数，样本量）)
    :return: 
    n_x: the size of the input layer
    n_y: the size of the output layer
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]

    return (n_x, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    initialize_parameters
    (参数初始化)
    :param n_x: size of the input layer 
    :param n_h: size of the hidden layer
    :param n_y: size of the output layer
    :return: 
    W1: weight matrix of shape (n_h, n_x) (第1层的权重矩阵(n_h, n_x))
    b1: bias vector of shape (n_h, 1) (第1层的偏移量向量(n_h, 1))
    W2: weight matrix of shape (n_y, n_h) (第2层的权重矩阵(n_y, n_h))
    b2: bias vector of shape (n_y, 1) (第2层的偏移量向量(n_y, 1))
    """
    # np.random.seed(2)  #Random initialization (随机种子初始化参数)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
    }

    return parameters


def forward_propagation(X, parameters):
    """
    forward_propagation
    (正向传播)
    :param X: input data of size (n_x, m)  (输入数据集X)
    :param parameters: python dictionary containing your parameters (output of initialization function) (字典类型， 权重以及偏移量参数)
    :return: 
    A2: The sigmoid output of the second activation (第2层激活函数sigmoid函数输出向量)
    cache: a dictionary containing "Z1", "A1", "Z2" and "A2" (字典类型,包含"Z1", "A1", "Z2", "A2")
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)            #第1层激活函数选择tanh
    Z2 = np.dot(W2, A1) + b2
    A2 = np.tanh(Z2)            #第2层激活函数选择tanh


    assert (A2.shape == (1, X.shape[1])) #若A2的大小和((1, X.shape[1])) 则直接报异常

    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2,
    }

    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    compute cost
    (计算成本函数)
    :param A2: The sigmoid output of the second activation, of shape (1, number of examples) (第2层激活函数sigmoid函数输出向量)
    :param Y: "true" labels vector of shape (1, number of examples) (正确标签向量)
    :param parameters: python dictionary containing your parameters W1, b1, W2 and b2 (字典类型，权重以及偏移量参数)
    :return: 
    cost: cross-entropy cost 
    """
    m = Y.shape[1]  # number of example

    W1 = parameters['W1']
    W2 = parameters['W2']

    logprobs = np.multiply(np.log(A2), Y)
    cost = - np.sum(np.multiply(np.log(A2), Y) + np.multiply(np.log(1. - A2), 1. - Y)) / m
    # cost = np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))/(-m)

    cost = np.squeeze(cost) #squeeze()函数的功能是：从矩阵shape中，去掉维度为1的。例如一个矩阵是的shape是（5， 1），使用过这个函数后，结果为（5，）。

    assert (isinstance(cost, float)) #若cost不是float型 则直接报异常

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    backward propagation
    (反向传播)
    :param parameters: python dictionary containing our parameters
    :param cache: a dictionary containing "Z1", "A1", "Z2" and "A2"
    :param X: input data of shape (2,number of examples)
    :param Y: "ture" labels vector of shape (1, number of examples)
    :return: 
    grads: python dictionary containing your gradients with respect to different parameters (字典类型，梯度微分参数)
    """
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - A1 ** 2)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2,
    }

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    update parameters
    (更新权重和偏移量参数)
    :param parameters: python dictionary containing your parameters
    :param grads: python dictionary containing your gradients 
    :param learning_rate (学习速率)
    :return: 
    :parameters:  python dictionary containing your updated parameters 
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }

    return parameters


def nn_model(X, Y, n_h, num_iterations, learning_rate, print_cost=False):
    """
    Forward Neural Network model
    (前向神经网络模型)
    :param X: input dataset of shape (input size, number of examples)  (输入数据集大小（几个属性，样本量）)
    :param Y: labels of shape (output size, number of exmaples) (标签数据大小（标签数，样本量）)
    :param n_h: size of the hidden layer (隐层神经元数量)
    :param num_iterations:  Number of iterations in gradient descent loop (迭代次数)
    :param learning_rate (学习速率)
    :param print_cost: if True, print the cost every 1000 iterations (是否打印显示)
    :return: 
    parameters: parameters learnt by the model. They can then be used to predict (训练完成后的参数)
    """

    # np.random.seed(4)
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[1]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    import pdb

    cost_list = []
    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)
        cost_list.append(cost)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters, cost_list


def predict(parameters, X):
    """
    predict
    (预测)
    :param paramters: python dictionary containing your parameters 
    :param X: input data of size (n_x, m)
    :return: 
    predictions: vector of predictions of our model
    """
    A2, cache = forward_propagation( X, parameters )
    #     Y_prediction = np.zeros((1, X.shape[1]))
    predictions = np.array([0 if i <= 0.5 else 1 for i in np.squeeze(A2)] )

    return predictions


def main():
    pass

if __name__ == '__main__':
    main()