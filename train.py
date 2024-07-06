import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def forward_propagation(X, parameters):
    activations = {'A0': X}
    L = len(parameters) // 2

    for l in range(1, L):
        Z = parameters['W' + str(l)].dot(activations['A' + str(l - 1)]) + parameters['b' + str(l)]
        activations['A' + str(l)] = 1 / (1 + np.exp(-Z)) # sigmoid activation

    ZL = parameters['W' + str(L)].dot(activations['A' + str(L - 1)]) + parameters['b' + str(L)]
    AL = 1 / (1 + np.exp(-ZL))
    activations['A' + str(L)] = AL

    return activations


def softmax(Z):
    exp_values = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
    return probabilities


def back_propagation(y, activation, parameters):
    m = y.shape[0]

    L = len(parameters) // 2
    dz = activation['A' + str(L)] - y.T

    gradients = {}
    for l in reversed(range(1, L + 1)):
        gradients['dW' + str(l)] = 1/m * np.dot(dz, activation['A' + str(l - 1)].T)
        gradients['db' + str(l)] = 1/m * np.sum(dz, axis=1, keepdims=True)
        if l > 1:
            dz = np.dot(parameters['W' + str(l)].T, dz) * activation['A' + str(l - 1)] * (1 - activation['A' + str(l - 1)])

    return gradients


def update_parameters(parameters, gradients, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * gradients['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * gradients['db' + str(l)]

    return parameters


def initialize_weights(dims):
    params = {}
    L = len(dims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(dims[l], dims[l - 1])
        params['b' + str(l)] = np.random.randn(dims[l], 1)

    return params


def train(X, y, nn):
    dims = np.insert(nn.hidden_layers, 0, nn.nb_features)
    dims = np.append(dims, 1)
    nn.parameters = initialize_weights(dims)
    params = nn.parameters

    # train_loss = []

    for i in tqdm(range(nn.epochs)):

        activations = forward_propagation(X, params)
        gradients = back_propagation(y, activations, params)
        params = update_parameters(params, gradients, nn.learning_rate)

        # y_pred = predict(X, params)
        # train_loss.append(log_loss(y, y_pred.T))

    nn.parameters = params
    y_pred = predict(X, params)

    # plt.plot(train_loss)
    # plt.xlabel('iterations')
    # plt.ylabel('cost')
    # plt.show()

    np.savetxt('datas/y_pred.csv', y_pred.T, fmt='%d', delimiter=',')
    return params


def predict(X, params):
    activations = forward_propagation(X, params)
    L = len(params) // 2
    Af = activations['A' + str(L)]
    return Af >= 0.5


def log_loss(y_true, y_pred):
    m = y_true.shape[0]
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -1 / m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
