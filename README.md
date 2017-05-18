This is a simple Feedforward neural network implemented in python with a command line tool, which might help getting to know how to use the NN class:

Example usage (XOR function):

``` sh
./fnn.py -n '[2,4,1]' -a tanh -o sigmoid -t '[[0,0,0],[0,1,1],[1,0,1],[1,1,0]]' -p '[[0,0],[0,1],[1,0],[1,1]]' -e 50000 -s model.nn
```

Where the FNN has 2 inputs, 4 neurons in one hidden layer and 1 output. -t means to train the following dataset (file or given string-array), -p means to predict the following values (file or values), -e the number of epochs, -s to save the model with the filename 'model.nn', -a to set the activation funtion of hidden layers as tanh and -o to set the activation funtion of the output layer as sigmoid.


Note: -n [2,4,4,1] would mean to use 2 hidden layers with 4 neurons each.

Outputs:
```
Dataset not a file; trying to parse string, e.g. '[[a1,b1,...],[a2,b2,...]]'
epochs: 0 ; error: 0.121040866385
epochs: 10000 ; error: 0.000441144231835
epochs: 20000 ; error: 0.00030402046664
epochs: 30000 ; error: 0.000153988194167
epochs: 40000 ; error: 0.000126763606198
Data to predict is not a file; trying to parse string, e.g. '[[a1,b1,...],[a2,b2,...]]'
[ 0.01289285]
[ 0.98902458]
[ 0.98899431]
[ 0.01414206]
```


```
Usage: fnn.py [options] dataset

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit
  -t NN_DATASET, --train=NN_DATASET
                        Train NN with provided dataset
  -n NN_SHAPE, --nnshape=NN_SHAPE
                        Set the neural network shape (inputs, hidden_layer1,
                        ..., outputs)  DEFAULT: '[2,1]'
  -p NN_PREDICT, --predict=NN_PREDICT
                        Predict using the model and given data
  -e NN_EPOCHS, --epochs=NN_EPOCHS
                        Set the number of epochs to train the NN  DEFAULT:
                        10000
  -a NN_ACTIVATION, --activation=NN_ACTIVATION
                        Set the activation function for the hidden layers
                        DEFAULT: tanh
  -o NN_OUTPUT_ACTIVATION, --outactivation=NN_OUTPUT_ACTIVATION
                        Set the activation function for the output layer
                        DEFAULT: sigmoid
  -l NN_LEARNING_RATE, --learningrate=NN_LEARNING_RATE
                        Set the learning rate for the NN   DEFAULT: 0.1
  -L NN_LAMBDA, --lambda=NN_LAMBDA
                        Set the regularization factor   DEFAULT: 0.0
  -A, --activfunctions  Show the activation functions
  -s MODEL_SAVE_FILENAME, --savemodel=MODEL_SAVE_FILENAME
                        Set filename to save the model after training
  -m MODEL_LOAD_FILENAME, --model=MODEL_LOAD_FILENAME
                        Load model to predict data or continue training it
  -d CSV_DELIMITER, --delimiter=CSV_DELIMITER
                        Set the delimiter of the CSV file   DEFAULT: ','
  -H, --hasheader       If set, the header (first line) of the dataset will be
                        omitted
  -I, --allinput        If set, the dataset will be read line by line instead
                        of random samples
  -r NN_REPORT_EVERY, --reportevery=NN_REPORT_EVERY
                        The number of epochs to report data from the trainig
                        process   DEFAULT: 10000

```
