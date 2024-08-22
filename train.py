import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


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

    print('X_train shape : ', X.shape)
    print('X_valid shape : ', X_val.shape)

    for i in tqdm(range(nn.epochs)):

        for j in range(0, X.shape[1], nn.batch_size):
            X_batch = X[:, j:j + nn.batch_size]
            y_batch = y[j:j + nn.batch_size]
            activations = forward_propagation(X_batch, params)
            gradients = back_propagation(y_batch, activations, params)
            params = update_parameters(params, gradients, nn.learning_rate)


        y_pred = predict(X, params)
        y_pred = to_one_hot(y_pred, 2)
        train_loss.append(compute_loss(y.T, y_pred))
        train_acc.append(compute_recall(y.T, y_pred))

        y_pred_val = predict(X_val, params)
        y_pred_val = to_one_hot(y_pred_val, 2)
        val_loss.append(compute_loss(y_val.T, y_pred_val))
        val_acc.append(compute_recall(y_val.T, y_pred_val))

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


def compute_recall(y_true, y_pred):
    true_positives = np.sum((y_true[0] == 1) & (y_pred[0] == 1))
    false_negatives = np.sum((y_true[0] == 0) & (y_pred[0] == 1))
    if (true_positives + false_negatives) == 0:
        return 0.0
    recall = true_positives / (true_positives + false_negatives)
    return recall



def predict(X, params):
    activations = forward_propagation(X, params)
    L = len(params) // 2
    Af = activations['A' + str(L)]
    predictions = np.argmax(Af, axis=0)
    return predictions


def to_one_hot(y, num_classes):
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
    correct_predictions = np.sum(y_true[0] == y_pred[0])
    total_predictions = y_true[0].shape[0]
    accuracy = correct_predictions / total_predictions
    return accuracy
