import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def forward_propagation(X, parameters):
    activations = {'A0': X}
    L = len(parameters) // 2

    for l in range(1, L + 1):
        Z = parameters['W' + str(l)].dot(activations['A' + str(l - 1)]) + parameters['b' + str(l)]
        activations['A' + str(l)] = 1 / (1 + np.exp(-Z))

    return activations


def back_propagation(y, activation, parameters):
    y = y.T
    m = y.shape[1]

    L = len(parameters) // 2
    dz = activation['A' + str(L - 1)] - y

    gradients = {}
    for l in reversed(range(1, L + 1)):
        gradients['dW' + str(l)] = 1/m * np.dot(dz, activation['A' + str(l - 1)].T)
        gradients['db' + str(l)] = 1/m * np.sum(dz, axis=1, keepdims=True)
        if l > 1:
            dz = np.dot(parameters['W' + str(l)], dz) * activation['A' + str(l - 1)] * (1 - activation['A' + str(l - 1)])

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
    X = X.T

    # print(nn)
    # for lay in nn.layers:
    # 	print(lay)

    dims = np.insert(nn.hidden_layers, 0, nn.nb_features)
    dims = np.append(dims, 1)
    nn.parameters = initialize_weights(dims)

    params = nn.parameters
    for key, val in params.items():
    	print(key, val.shape)

    train_loss = []
    train_acc = []

    for i in tqdm(range(nn.epochs)):

        activations = forward_propagation(X, params)
        # for key, val in activations.items():
        # 	print(key, val.shape)
        gradients = back_propagation(y, activations, params)
        # for key, val in gradients.items():
        # 	print(key, val.shape)

        params = update_parameters(params, gradients, nn.learning_rate)
        print('===============')
        for key, val in params.items():
            print(key, val.shape)

        if i % 10 == 0:
            L = len(params) // 2
            # train_loss.append(log_loss(y, activations['A' + str(L)])

    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    # ax[0].plot(range(nn.epochs), train_loss)