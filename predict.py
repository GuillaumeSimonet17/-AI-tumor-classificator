import train
import argparse
import numpy as np
import pandas as pd


def predict(X, params):
    activations = train.forward_propagation(X, params)
    L = len(params) // 2
    Af = activations['A' + str(L)]
    return Af >= 0.5


def load_model(file_path):
    data = np.load(file_path)
    parameters = {key: data[key] for key in data.files}
    return parameters


def standardization(data_to_std):
    return (data_to_std - np.mean(data_to_std, axis=0)) / np.std(data_to_std, axis=0)


def compute_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = y_true.shape[0]
    accuracy = correct_predictions / total_predictions
    return accuracy


def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -1 / m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


if __name__ == '__main__':
    test_features = pd.read_csv('datas/test.csv', header=None)
    type_of_tumor = test_features.iloc[:,1]
    data = test_features.iloc[:,2:]
    test_features_std = standardization(data)
    params = load_model('datas/params.npz')
    test_Y_bool = [1 if x == 'M' else 0 for x in type_of_tumor]
    test_predict = predict(test_features_std.T, params)
    acc = compute_accuracy(np.array(test_Y_bool), test_predict)
    loss = compute_loss(np.array(test_Y_bool), test_predict)
    print('Accuracy on total set =', acc)
    print('Loss on total set =', loss)
    print('---------------------------------')


    test_features = pd.read_csv('datas/validation_X_std.csv', header=None)
    test_bools = pd.read_csv('datas/validation_Y_bool.csv', header=None)
    params = load_model('datas/params.npz')
    test_predict = predict(test_features.T, params)
    acc = compute_accuracy(np.array(test_bools).flatten(), test_predict)
    loss = compute_loss(np.array(test_bools).flatten(), test_predict)
    print('Accuracy on validation set =', acc)
    print('Loss on validation set =', loss)
    print('---------------------------------')


    test_features = pd.read_csv('datas/train_X_std.csv', header=None)
    test_bools = pd.read_csv('datas/train_Y_bool.csv', header=None)
    test_features_std = standardization(test_features)
    params = load_model('datas/params.npz')
    test_predict = predict(test_features_std.T, params)
    acc = compute_accuracy(np.array(test_bools).flatten(), test_predict)
    loss = compute_loss(np.array(test_bools).flatten(), test_predict)
    print('Accuracy on train set =', acc)
    print('Loss on train set =', loss)