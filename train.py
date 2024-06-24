import argparse
import numpy as np
import pandas as pd


class NeuralNetwork:
	def __init__(self, layers, dims, epochs, batch_size, learning_rate, loss, nb_features):
		self.layers = layers
		self.epochs = epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.loss = loss
		total_layers = np.insert(dims, 0, nb_features)
		total_layers = np.append(total_layers, 1)
		print(total_layers)
		self.parameters = initialize_weights(total_layers)
		for key, val in self.parameters.items():
			print(key, val.shape)

	def __str__(self):
		return (f'epochs: {self.epochs}, '
				f'batch_size: {self.batch_size}, '
				f'learning_rate: {self.learning_rate} ')


def initialize_weights(dims):
	params = {}
	layers = len(dims)

	for l in range(layers):
		params['W' + str(l)] = np.random.randn(dims[l], dims[l - 1])
		params['b' + str(l)] = np.random.randn(dims[l], 1)

	return params


class Layer:
	def __init__(self, size, nb_units, activation):
		self.input_size = size
		self.nb_units = nb_units
		self.activation = activation

	def __str__(self):
		return f'input_size: {self.input_size}, nb_units: {self.nb_units}, activation: {self.activation}'


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
	layers = []

	for layer in args.layers:
		layers.append(Layer(
			nb_features,
			layer,
			args.activation,
		))
	layers.append(Layer(
		nb_features,
		1,
		'softmax',
	))

	return NeuralNetwork(layers, args.layers, args.epochs, args.batch_size, args.learning_rate, args.loss, nb_features)


if __name__ == '__main__':
	args = parse_args()
	features = pd.read_csv('train_X_std.csv')
	neural_network = create_neural_network(args, features.shape[1])

	print(neural_network)
	for lay in neural_network.layers:
		print(lay)
