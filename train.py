import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import pandas as pd
from classes import NeuralNetwork


def forward_propagation(X, parameters):
    activations = {'A0': X}
    L = len(parameters) // 2

    for l in range(1, L):
        Z = parameters['W' + str(l)].dot(activations['A' + str(l - 1)]) + parameters['b' + str(l)]
        activations['A' + str(l)] = 1 / (1 + np.exp(-Z))

    ZL = parameters['W' + str(L)].dot(activations['A' + str(L - 1)]) + parameters['b' + str(L)]
    AL = softmax(ZL)
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
            df = activation['A' + str(l - 1)] * (1 - activation['A' + str(l - 1)])
            dz = np.dot(parameters['W' + str(l)].T, dz) * df

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
        params['b' + str(l)] = np.zeros((dims[l], 1))
    return params


def train(X, X_val, y, y_val, nn):
    dims = np.insert(nn.hidden_layers, 0, nn.nb_features)
    dims = np.append(dims, 2)
    nn.parameters = initialize_weights(dims)
    params = nn.parameters

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    print('X_train shape : ', X.shape[1])
    print('X_valid shape : ', X_val.shape)
    print('y_train shape : ', y.shape)
    print('y_valid shape : ', y_val.shape)

    for i in tqdm(range(nn.epochs)):
        for j in range(0, X.shape[1], nn.batch_size):
            X_batch = X[:, j:j + nn.batch_size]
            y_batch = y.T[j:j + nn.batch_size]

            activations = forward_propagation(X_batch, params)
            gradients = back_propagation(y_batch, activations, params)
            params = update_parameters(params, gradients, nn.learning_rate)

        y_pred = predict(X, params)
        y_pred = to_one_hot(y_pred.T, 2)
        y_pred = y_pred[:, ::-1]
        train_loss.append(compute_loss(y, y_pred.T))
        train_acc.append(compute_recall(y, y_pred.T))


        y_pred_val = predict(X_val, params)
        y_pred_val = to_one_hot(y_pred_val.T, 2)
        y_pred_val = y_pred_val[:, ::-1]
        val_loss.append(compute_loss(y_val, y_pred_val.T))
        val_acc.append(compute_recall(y_val, y_pred_val.T))

        print(f'epoch {i}/{nn.epochs} - loss: {train_loss[i]} - val_loss: {val_loss[i]}')

    nn.parameters = params
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(range(nn.epochs), train_loss, label='Training Loss')
    ax[0].plot(range(nn.epochs), val_loss, label='Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(range(nn.epochs), train_acc, label='Training Accuracy')
    ax[1].plot(range(nn.epochs), val_acc, label='Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy')
    ax[1].legend()

    plt.show()
    # np.savetxt('datas/y_pred.csv', y_pred.T, fmt='%d', delimiter=',')
    return params


def predict(X, params):
    activations = forward_propagation(X, params)
    L = len(params) // 2
    Af = activations['A' + str(L)]
    predictions = np.argmax(Af, axis=0)
    return predictions


def to_one_hot(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y.flatten()] = 1
    return one_hot[:, ::-1]


def to_one_hot1(y, num_classes):
    one_hot = np.zeros((num_classes, y.shape[0]))
    one_hot[y.flatten(), np.arange(y.shape[0])] = 1
    return one_hot


def compute_loss(y_true, y_pred):
    m = y_true.shape[1]  # y_true et y_pred doivent être de forme (n_classes, m)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -1 / m * np.sum(y_true * np.log(y_pred))
    return loss


def compute_accuracy(y_true, y_pred):
    y_true_col1 = y_true.T[:, 0]
    y_pred_col1 = y_pred.T[:, 0]
    correct_predictions = np.sum(y_true_col1 == y_pred_col1)
    total_predictions = y_true.shape[1]
    return correct_predictions / total_predictions


def compute_recall(y_true, y_pred):
    y_true_col1 = y_true.T[:, 0]
    y_pred_col1 = y_pred.T[:, 0]
    true_positives = np.sum((y_true_col1 == 1) & (y_pred_col1 == 1))
    false_negatives = np.sum((y_true_col1 == 1) & (y_pred_col1 == 0))
    # print('true_positives = ', true_positives)
    # print('false_negatives = ', false_negatives)
    # print('false_positives = ', false_positives)
    if (true_positives + false_negatives) == 0:
        return 0.0
    recall = true_positives / (true_positives + false_negatives)
    return recall


def compute_precision(y_true, y_pred):
    y_true_col1 = y_true.T[:, 0]
    y_pred_col1 = y_pred.T[:, 0]
    true_positives = np.sum((y_true_col1 == 1) & (y_pred_col1 == 1))
    false_positives = np.sum((y_true_col1[0] == 0) & (y_pred_col1[0] == 1))
    if (true_positives + false_positives) == 0:
        return 0.0
    precision = true_positives / (true_positives + false_positives)
    return precision

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', nargs='+', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--loss', type=str, default='binary_crossentropy')
    parser.add_argument('--activation', type=str, default='sigmoid')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_features = pd.read_csv('datas/train_X_std.csv', header=None)
    train_y = np.array(pd.read_csv('datas/train_Y_bool.csv', header=None)).reshape(-1, 1)
    train_y = to_one_hot(train_y, 2)

    val_features = pd.read_csv('datas/validation_X_std.csv', header=None)
    val_y = np.array(pd.read_csv('datas/validation_Y_bool.csv', header=None)).reshape(-1, 1)
    val_y = to_one_hot(val_y, 2)

    neural_network = NeuralNetwork(args.layers, args.epochs, args.batch_size, args.learning_rate, args.loss, train_features.shape[1])
    if neural_network.batch_size > train_features.shape[1]:
        raise ValueError('Batch size can\'t be larger than the number of features')

    X = np.array(train_features)
    X_val = np.array(val_features)

    params = train(X.T, X_val.T, train_y.T, val_y.T, neural_network)

    np.savez('datas/params', **params)


# 1 - split program use seed
# 2 - training program print perf during training at each epoch
# 3 - at the end of training, plot loss and acc for train and valid
# 4 - predict take one exemple and calcul error function


# TODO : Assurez-vous que les dimensions des matrices et vecteurs sont correctes tout au long des opérations. Cela est particulièrement important pour le produit matriciel et les opérations de diffusion.
# TODO : Verifier les load des fichiers (si vides...)