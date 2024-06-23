import numpy as np
import argparse

class NeuralNetwork:
	def __init__(self, layers):
		self.layers = layers


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--layers', nargs='+', type=int, required=True)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--learning_rate', type=float, default=0.01)
	parser.add_argument('--loss', type=str, default='binary_crossentropy')
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	print(args)
