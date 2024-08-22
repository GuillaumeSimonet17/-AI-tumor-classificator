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


def to_one_hot(y, num_classes):
    one_hot = np.zeros((num_classes, y.shape[0]))
    one_hot[y.flatten(), np.arange(y.shape[0])] = 1
    return one_hot


if __name__ == '__main__':
    args = parse_args()
    train_features = pd.read_csv('datas/train_X_std.csv', header=None)
    train_y = np.array(pd.read_csv('datas/train_Y_bool.csv', header=None)).reshape(-1, 1)
    train_y = to_one_hot(train_y, 2).T

    val_features = pd.read_csv('datas/validation_X_std.csv', header=None)
    val_y = np.array(pd.read_csv('datas/validation_Y_bool.csv', header=None)).reshape(-1, 1)
    val_y = to_one_hot(val_y, 2).T

    neural_network = NeuralNetwork(args.layers, args.epochs, args.batch_size, args.learning_rate, args.loss, train_features.shape[1])
    if neural_network.batch_size > train_features.shape[1]:
        raise ValueError('Batch size can\'t be larger than the number of features')

    X = np.array(train_features)
    X_val = np.array(val_features)
    params = train.train(X.T, X_val.T, train_y, val_y, neural_network)

    # pred = train.predict(X.T, params)

    np.savez('datas/params', **params)


# 1 - split program use seed
# 2 - training program print perf during training at each epoch
# 3 - at the end of training, plot loss and acc for train and valid
# 4 - predict take one exemple and calcul error function

# le set de validation est juste pour calculer l'accuracy

# TODO : Assurez-vous que les dimensions des matrices et vecteurs sont correctes tout au long des opérations. Cela est particulièrement important pour le produit matriciel et les opérations de diffusion.
# TODO : Verifier les load des fichiers (si vides...)
# TODO : faire softmax
