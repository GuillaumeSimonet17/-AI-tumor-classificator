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


def create_neural_network(args, nb_features):
    # layers = []
    #
    # for nb_units in args.layers:
    #     layers.append(Layer(
    #         nb_features,
    #         nb_units,
    #         args.activation,
    #     ))
    # layers.append(Layer(
    #     nb_features,
    #     1,
    #     'softmax',
    # ))

    return NeuralNetwork(args.layers, args.epochs, args.batch_size, args.learning_rate, args.loss, nb_features)


if __name__ == '__main__':
    args = parse_args()
    features = pd.read_csv('datas/train_X_std.csv', header=None)
    y = np.array(pd.read_csv('datas/train_Y_bool.csv', header=None)).reshape(-1, 1)

    neural_network = create_neural_network(args, features.shape[1])

    train.train(features.T, y, neural_network)

    # validation_features = pd.read_csv('datas/validation_X_std.csv', header=None)
    # validation_predict = train.predict(validation_features.T, neural_network.parameters)
    # validation_y = np.array(pd.read_csv('datas/validation_Y_bool.csv', header=None)).reshape(-1, 1)
    #
    # np.savetxt('datas/validation_predict.csv', validation_predict.T, fmt='%d', delimiter=',')

# TODO: se servrir de layer car je m'en sert pas
# TODO: mettre params from trained dans un fichier et les recup pour predict
# TODO: check comment se servir de loss et batch_size
# TODO: weights_initializer='heUniform')
# TODO: mettre predict dans un fichier
# TODO: training/validation accuracy plot