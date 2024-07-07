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
    if neural_network.batch_size > features.shape[1]:
        raise ValueError('Batch size can\'t be larger than the number of features')

    X = np.array(features)
    params = train.train(X.T, y, neural_network)
    np.savez('datas/params', **params)


# 1 - split programm use seed
# 2 - training programm print perf during training at each epoch
# 3 - at the end of training, plot loss and acc for train and valid
# 4 - predict take one exemple and calcul error function

# TODO: To visualize your model performances during training, you will display at each epoch
# the training and validation metrics.
# x_train shape : (342, 30)
# x_valid shape : (85, 30)
# epoch 01/70 - loss: 0.6882 - val_loss: 0.6788

# Enfaite, dans predict j'ai mis validation data mais, ce program doit attendre qu'un exemple,
# pas une centaine..
# Enfaite, validation sert dans le training :
# y_pred_val = predict(X_val, params)
#         val_loss.append(log_loss(y_val, y_pred_val.T))
#         val_accuracy.append(calculate_accuracy(y_val, y_pred_val.T))
# c'est tout !

# TODO: Donc faut que je refasse mon predict.py pour faire en sorte qu'il predise pour un seul exemple
# TODO: le predict.py doit evaluer la prediction using the binary cross-entropy error function.

# TODO: pour split je dois utiliser seed pour avoir les mêmes resultats
# TODO: training/validation accuracy plot (not really what I expected)

# TODO: weights_initializer='heUniform')
# TODO: faire softmax
