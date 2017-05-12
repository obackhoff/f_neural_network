import numpy as np
import pickle


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(fx):
    return fx*(1.0 - fx)

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_prime(x):
    return (np.exp(x) - 1) / np.exp(x)

def elu(x):
    a = 0.5
    return np.where(x < 0, a*(np.exp(x) - 1), x)

def elu_prime(fx):
    a = 0.5
    return np.where(fx < 0, (fx + a), 1)

def tanh(x):
    return np.tanh(1*x)

def tanh_prime(fx):
    return 1*(1.0 - (fx * fx))

def lin(x):
    return x

def lin_prime(fx):
    return 1.0


class NeuralNetwork:

    def __init__(self, layers, activation='sigmoid', output_activation='sigmoid'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
        elif activation == 'softplus':
            self.activation = softplus
            self.activation_prime = softplus_prime
        elif activation == 'elu':
            self.activation = elu
            self.activation_prime = elu_prime

        if output_activation == 'sigmoid':
            self.output_activation = sigmoid
            self.output_activation_prime = sigmoid_prime
        elif output_activation == 'tanh':
            self.output_activation = tanh
            self.output_activation_prime = tanh_prime
        elif output_activation == 'softplus':
            self.output_activation = softplus
            self.output_activation_prime = softplus_prime
        elif output_activation == 'elu':
            self.output_activation = elu
            self.output_activation_prime = elu_prime
        elif output_activation == 'lin':
            self.output_activation = lin
            self.output_activation_prime = lin_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        i = 0
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1.0
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2 * np.random.random( (layers[i] + 1, layers[i+1])) - 1.0
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000, all_input=False):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)

        if all_input:
            i = -1

        for k in range(epochs):

            if all_input:
                i += 1
                i = i % X.shape[0]
            else:
                i = np.random.randint(X.shape[0])

            a = [X[i]]

            # Forward propagation begins
            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    if(l == len(self.weights) - 1):
                        activation = self.output_activation(dot_value)
                    else:
                        activation = self.activation(dot_value)
                    a.append(activation)

            # Backpropagation begins      
            # First the output layer
            error = (y[i] - a[-1]) #* (y[i] - a[-1]) * 0.5
            deltas = [error * self.output_activation_prime(a[-1])]


            # then the from second to last layer to the first hidden layer
            for l in range(len(a) - 2, 0, -1):
                dot_prod = deltas[-1].dot(self.weights[l].T)
                deltas.append(dot_prod * self.activation_prime(a[l]))

            # reverse layers to have them in order again
            deltas.reverse()

            # backpropagation:
            # - Multiply its output delta and input activation
            #    to get the gradient of the weight.
            # - Subtract a ratio (percentage) of the gradient from the weight.
            for j in range(len(self.weights)):
                layer = np.atleast_2d(a[j])
                delta = np.atleast_2d(deltas[j])
                self.weights[j] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0:
                print('epochs: ' + str(k), '; error: ' + str((error*error*0.5).sum()/error.shape[0]))

    def predict(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)):
            if(l == len(self.weights) - 1):
                a = self.output_activation(np.dot(a, self.weights[l]))
            else:
                a = self.activation(np.dot(a, self.weights[l]))
        return a

    def save_model(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load_model(filename):
        with open(filename, 'rb') as input:
            return pickle.load(input)

if __name__ == '__main__':

    pass
