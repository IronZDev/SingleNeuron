import numpy as np


# noinspection PyMethodMayBeStatic
class Neuron:
    def __init__(self, activation='heaviside', learning_rate=0.01):
        # seeding for random number generation
        np.random.seed(1)
        self.weights = np.random.uniform(0, 1, 3)
        self.learning_rate = learning_rate
        self.activation_function = getattr(self, activation)

    def sigmoid(self, x, derivative=False):
        if not derivative:
            return 1 / (1 + np.exp(-x))
        else:
            return self.sigmoid(x) * (1 - self.sigmoid(x))

    def heaviside(self, x, derivative=False):
        if not derivative:
            return np.heaviside(x, 1)
        else:  # Derivative mode
            return 1

    def sin(self, x, derivative=False):
        if not derivative:
            return np.sin(x)
        else:
            return np.cos(x)

    def cos(self, x, derivative=False):
        if not derivative:
            return np.cos(x)
        else:
            return -np.sin(x)

    def tanh(self, x, derivative=False):
        if not derivative:
            return np.tanh(x)
        else:
            return 1 - np.tanh(x)**2

    def sign(self, x, derivative=False):
        if not derivative:
            if x < 0:
                return -1
            elif x > 0:
                return 1
            else:
                return 0
        else:
            return 1

    def relu(self, x, derivative=False):
        if not derivative:
            if x > 0:
                return x
            else:
                return 0
        else:
            if x > 0:
                return 1
            else:
                return 0

    def leaky_relu(self, x, derivative=False):
        if not derivative:
            if x > 0:
                return x
            else:
                return 0.01 * x
        else:
            if x > 0:
                return 1
            else:
                return 0.01

    def train(self, training_inputs, training_outputs, training_iterations):
        for _ in range(training_iterations):
            for input_val, output in zip(training_inputs, training_outputs):
                calculated = self.predict(self.weights @ input_val)
                error = output - calculated
                adjustments = self.learning_rate * error * self.activation_function(self.weights @ input_val, True) * input_val
                self.weights += adjustments

    def predict(self, ins):
        # passing the inputs via the neuron to get output
        ins = ins.astype(float)
        output = self.activation_function(ins)
        return output
