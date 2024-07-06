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


if __name__ == '__main__':
    validation_features = pd.read_csv('datas/validation_X_std.csv', header=None)
    params = load_model('datas/params.npz')

    validation_predict = predict(validation_features.T, params)

    np.savetxt('datas/validation_predict.csv', validation_predict.T, fmt='%d', delimiter=',')
