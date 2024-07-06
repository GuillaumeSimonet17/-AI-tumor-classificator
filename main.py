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

    neural_network = NeuralNetwork(args.layers, args.epochs, args.batch_size, args.learning_rate, args.loss, features.shape[1])

    params = train.train(features.T, y, neural_network)
    np.savez('datas/params', **params)

# TODO: check comment se servir de loss et batch_size
# TODO: weights_initializer='heUniform')
# TODO: training/validation accuracy plot
