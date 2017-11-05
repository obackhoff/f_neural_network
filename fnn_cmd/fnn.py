#!/bin/env python
from optparse import OptionParser
import json
import numpy as np
import pickle
import os.path

ACTIVATIONS = ["sigmoid", "softplus", "elu", "relu", "tanh"]
OUT_ACTIVATIONS = ["lin","sigmoid", "softplus", "elu", "relu", "tanh"]

# DEFINIG ACTIVATION FUNCTIONS
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(fx): 
    return fx*(1.0 - fx)

def gauss(x):
    return np.exp(-(x**2))

def gauss_prime(x): 
    return -2*x*gauss(x)

def softmax(x):
    exps = np.exp(x)
    return exps / exps.sum()

def softmax_prime(fx):
    #s = fx.reshape(-1,1)
    return fx*(1 - fx)#np.diagflat(s) - np.dot(s, s.T)

def fastsig(x):
    return x/(1.0 + np.abs(x))

def fastsig_prime(x):
    return 1.0 / (1 + np.abs(x))

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_prime(x):
    return (np.exp(x) - 1) / np.exp(x)

def elu(x):
    a = 1
    return np.where(x < 0, a*(np.exp(x) - 1), x)

def elu_prime(fx):
    a = 1
    return np.where(fx < 0, (fx + a), 1)

def relu(x):
    return np.where(x < 0, 0, x)

def relu_prime(fx):
    return np.where(fx < 0, 0, 1)

def tanh(x):
    return np.tanh(1*x)

def tanh_prime(fx):
    return 1*(1.0 - (fx * fx))

def lin(x):
    return x

def lin_prime(fx):
    return 1.0


# FeedForward Neural Network
class FNN:

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
        elif activation == 'relu':
            self.activation = relu
            self.activation_prime = relu_prime
        elif activation == 'fastsig':
            self.activation = fastsig
            self.activation_prime = fastsig_prime

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
        elif output_activation == 'relu':
            self.output_activation = relu
            self.output_activation_prime = relu_prime
        elif output_activation == 'fastsig':
            self.output_activation = fastsig
            self.output_activation_prime = fastsig_prime
        elif output_activation == 'softmax':
            self.output_activation = softmax
            self.output_activation_prime = softmax_prime
        elif output_activation == 'gauss':
            self.output_activation = gauss
            self.output_activation_prime = gauss_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        self.layers = layers
        self.loss_func = []
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1.0
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2 * np.random.random( (layers[i] + 1, layers[i+1])) - 1.0
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=50000, report_every=10000, lmbda=0, batch_size=1):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
       
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        cnt = 0
        error_avg = 0
        for k in range(epochs):
            p = np.random.permutation(X.shape[0])
            X = X[p]
            y = y[p]
            X_batches = np.array(np.array_split(X, X.shape[0]/batch_size))
            y_batches = np.array(np.array_split(y, y.shape[0]/batch_size))           
            for b_idx in range(len(y_batches)):               
                grads = [0] * len(self.weights)
                for x_idx in range(len(y_batches[0])):
                    a = [X_batches[b_idx][x_idx]]
                    cnt += 1
                     # Forward propagation begins
                    for l in range(len(self.weights)):
                        dot_value = np.dot(a[l], self.weights[l])
                        if(l == len(self.weights) - 1):
                            activation = self.output_activation(dot_value)
                        else:
                            activation = self.activation(dot_value)
                        a.append(activation)
                    
                    error = (y_batches[b_idx][x_idx] - a[-1]) #* (y[i] - a[-1]) * 0.5
                    norm_err = np.linalg.norm(error)
                    error_avg += norm_err
                    

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
                        grads[j] += layer.T.dot(np.atleast_2d(deltas[j]))
        
                    
                self.loss_func.append(error_avg/cnt)   
                for j in range(len(self.weights)):
                    self.weights[j] = (1 - lmbda*learning_rate/X.shape[0])*(self.weights[j]) + \
                    learning_rate/batch_size * grads[j]
#                    self.weights[j] += -learning_rate * (grads[j]/batch_size + lmbda/X.shape[0] * \
#                    self.weights[j])
                
                if cnt % report_every == 0:
                    print('epoch: ' + str(k+1), '; error: ' + str(error_avg/cnt))
                    
        print('Average error: ' + str(error_avg/cnt))

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


# Helper functions
def show_activations():
    print("Activations: \n")
    for a in ACTIVATIONS:
        print(a)
    print("\nOutput activations: \n")
    for a in OUT_ACTIVATIONS:
        print(a)

def run():
    parser = OptionParser(usage="usage: %prog [options] dataset",
                          version="%prog 1.0")
    parser.add_option("-t", "--train",
                      dest="nn_dataset",
                      default=None,
                      help="Train NN with provided dataset")
    parser.add_option("-n", "--nnshape",
                      dest="nn_shape",
                      default=[2,1],
                      help="Set the neural network shape (inputs, hidden_layer1, ..., outputs)\n DEFAULT: '[2,1]'")
    parser.add_option("-p", "--predict",
                      dest="nn_predict",
                      default=None,
                      help="Predict using the model and given data")
    parser.add_option("-e", "--epochs",
                      dest="nn_epochs",
                      default=30000,
                      help="Set the number of epochs to train the NN\n DEFAULT: 10000")
    parser.add_option("-a", "--activation",
                      dest="nn_activation",
                      default="tanh",
                      help="Set the activation function for the hidden layers \n DEFAULT: tanh")
    parser.add_option("-o", "--outactivation",
                      dest="nn_output_activation",
                      default="sigmoid",
                      help="Set the activation function for the output layer\n DEFAULT: sigmoid")
    parser.add_option("-l", "--learningrate",
                      dest="nn_learning_rate",
                      default=0.1,
                      help="Set the learning rate for the NN \n DEFAULT: 0.1")
    parser.add_option("-L", "--lambda",
                      dest="nn_lambda",
                      default=0.0,
                      help="Set the regularization factor \n DEFAULT: 0.0")
    parser.add_option("-A", "--activfunctions",
                      action="store_true",
                      dest="show_activ",
                      default=False,
                      help="Show the activation functions")
    parser.add_option("-s", "--savemodel",
                      dest="model_save_filename",
                      default="",
                      help="Set filename to save the model after training")
    parser.add_option("-m", "--model",
                      dest="model_load_filename",
                      default="",
                      help="Load model to predict data or continue training it")
    parser.add_option("-d", "--delimiter",
                      dest="csv_delimiter",
                      default=",",
                      help="Set the delimiter of the CSV file \n DEFAULT: ','")
    parser.add_option("-H", "--hasheader",
                      action="store_true",
                      dest="csv_hasheader",
                      default=False,
                      help="If set, the header (first line) of the dataset will be omitted")
    parser.add_option("-b", "--batch-size",
                      dest="nn_bs",
                      default=1,
                      help="If set > 1, the NN is trained using Mini-Batch GD")
    parser.add_option("-r", "--reportevery",
                      dest="nn_report_every",
                      default=10000,
                      help="The number of epochs to report data from the trainig process \n DEFAULT: 10000")
    parser.add_option("-Q", "--quiet",
                      action="store_true",
                      dest="nn_donotprint",
                      default=False,
                      help="If set, program will not print the predictions")



    (options, args) = parser.parse_args()
    predictions = []
    if  options.nn_dataset == None and options.nn_predict == None and options.show_activ == False:
        parser.error("use -h to get help")

    if options.show_activ:
        show_activations()

    nn = None
    if not options.model_load_filename == "":
        nn = FNN.load_model(options.model_load_filename)

    if not options.nn_dataset == None:
        dataset = np.array([])
        isFile = os.path.isfile(options.nn_dataset)
        if isFile:
            dataset = np.genfromtxt(options.nn_dataset, delimiter=options.csv_delimiter)
            if options.csv_hasheader:
                dataset = dataset[1:]
        else:
            print("Dataset not a file; trying to parse string, e.g. '[[a1,b1,...],[a2,b2,...]]'")
            dataset = np.array(json.loads(options.nn_dataset))
            if options.csv_hasheader:
                dataset = dataset[1:]

        nnShape = list(json.loads(options.nn_shape))
        outputIndex = int(nnShape[-1])
        if nn == None:
            nn = FNN(nnShape,
                activation=options.nn_activation,
                output_activation=options.nn_output_activation)

        outputs = dataset[ : , -outputIndex : ]
        inputs = dataset[ : , 0 : -outputIndex]

        nn.fit(inputs, outputs,
            learning_rate=float(options.nn_learning_rate),
            epochs=int(options.nn_epochs),
            report_every=int(options.nn_report_every),
            lmbda=float(options.nn_lambda),
            batch_size=int(options.nn_bs))

        if not options.model_save_filename == "":
            nn.save_model(options.model_save_filename)

    if not options.nn_predict == None:
        isFile = os.path.isfile(options.nn_predict)
        if isFile:
            options.nn_predict = np.genfromtxt(options.nn_predict, delimiter=options.csv_delimiter)
        else:
            print("Data to predict is not a file; trying to parse string, e.g. '[[a1,b1,...],[a2,b2,...]]'")
            options.nn_predict = np.array(json.loads(options.nn_predict))
        for x in options.nn_predict:
            p = nn.predict(x);
            predictions.append(p)
            if not options.nn_donotprint:
                print(p)
            
    return np.array(predictions)


# Command line option parser and executor
if __name__ == '__main__':
    run()
