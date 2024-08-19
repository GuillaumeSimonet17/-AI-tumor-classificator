import argparse
import numpy as np
import pandas as pd
import train
from classes import NeuralNetwork

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
    features = pd.read_csv('datas/train_X_std.csv', header=None)
    y = np.array(pd.read_csv('datas/train_Y_bool.csv', header=None)).reshape(-1, 1)

    features_val = pd.read_csv('datas/validation_X_std.csv', header=None)
    y_val = np.array(pd.read_csv('datas/validation_Y_bool.csv', header=None)).reshape(-1, 1)

    neural_network = NeuralNetwork(args.layers, args.epochs, args.batch_size, args.learning_rate, args.loss, features.shape[1])
    if neural_network.batch_size > features.shape[1]:
        raise ValueError('Batch size can\'t be larger than the number of features')

    X = np.array(features)
    X_val = np.array(features_val)
    params = train.train(X.T, X_val.T, y, y_val, neural_network)

    val_pred = train.predict(X_val.T, params)
    val_loss = train.compute_loss(y_val.flatten(), val_pred)

    pred = train.predict(X.T, params)
    loss = train.compute_loss(y.flatten(), pred)

    print(val_loss - loss)
    print(loss)
    print('---------------')

    for i in range(10):
        if loss > 1.7 and val_loss - loss > 0.7:
            neural_network.epochs += 10
            params = train.train(X.T, X_val.T, y, y_val, neural_network)

            val_pred = train.predict(X_val.T, params)
            val_loss = train.compute_loss(y_val.flatten(), val_pred)

            pred = train.predict(X.T, params)
            loss = train.compute_loss(y.flatten(), pred)
            print(val_loss - loss)
            print(loss)
            print('---------------')
        else:
            break

    np.savez('datas/params', **params)


# 1 - split program use seed
# 2 - training program print perf during training at each epoch
# 3 - at the end of training, plot loss and acc for train and valid
# 4 - predict take one exemple and calcul error function

# TODO : Assurez-vous que les dimensions des matrices et vecteurs sont correctes tout au long des opérations. Cela est particulièrement important pour le produit matriciel et les opérations de diffusion.
# TODO : Verifier les load des fichiers (si vides...)
# TODO: faire softmax
