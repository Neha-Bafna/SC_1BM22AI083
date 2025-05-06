#1
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.lr = learning_rate
        self.epochs = epochs

    def activate(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.append(x, 1)
        return self.activate(np.dot(self.weights, x))

    def train(self, X, y):
        for _ in range(self.epochs):
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                error = target - prediction

                self.weights[:-1] += self.lr * error * inputs
                self.weights[-1] += self.lr * error


if __name__ == "__main__":

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    perceptron = Perceptron(input_size=2)
    perceptron.train(X, y)

    for inputs in X:
        print(f"Input: {inputs}, Output: {perceptron.predict(inputs)}")

#2
import numpy as np

class HebbianNN:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0

    def activate(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        return self.activate(np.dot(self.weights, x) + self.bias)

    def train(self, X, y):
        for inputs, target in zip(X, y):
            self.weights += inputs * target
            self.bias += target

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    nn = HebbianNN(input_size=2)
    nn.train(X, y)

    for inputs in X:
        print(f"Input: {inputs}, Output: {nn.predict(inputs)}")
