
class NeuralNetwork:
    def __init__(self, hidden_layers, epochs, batch_size, learning_rate, loss, nb_features):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.hidden_layers = hidden_layers
        self.nb_features = nb_features
        self.parameters = False

    def __str__(self):
        return (f'epochs: {self.epochs}, '
                f'batch_size: {self.batch_size}, '
                f'learning_rate: {self.learning_rate}, '
                f'loss: {self.loss}, '
                f'hidden_layers: {self.hidden_layers}, '
                f'nb_features: {self.nb_features} '
        )
