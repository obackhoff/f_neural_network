This is a simple Feedforward neural network implemented in python with a command line tool, which might help geting to know how to use the NN class:

```
Usage: fnn.py [options] dataset

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit
  -t NN_DATASET, --train=NN_DATASET
                        Train NN with provided dataset
  -n NN_SHAPE, --nnshape=NN_SHAPE
                        Set the neural network shape (inputs, hidden_layer1,
                        ..., outputs)
  -p NN_PREDICT, --predict=NN_PREDICT
                        Predict using the model and given data
  -e NN_EPOCHS, --epochs=NN_EPOCHS
                        Set the number of epochs to train the NN
  -a NN_ACTIVATION, --activation=NN_ACTIVATION
                        Set the activation function for the hidden layers
  -o NN_OUTPUT_ACTIVATION, --outactivation=NN_OUTPUT_ACTIVATION
                        Set the activation function for the output layer
  -l NN_LEARNING_RATE, --learningrate=NN_LEARNING_RATE
                        Set the learning rate for the NN
  -L NN_LAMBDA, --lambda=NN_LAMBDA
                        Set the regularization factor
  -A, --activfunctions  Show the activation functions
  -s MODEL_SAVE_FILENAME, --savemodel=MODEL_SAVE_FILENAME
                        Set filename to save the model after training
  -m MODEL_LOAD_FILENAME, --model=MODEL_LOAD_FILENAME
                        Load model to predict data or continue training it
  -d CSV_DELIMITER, --delimiter=CSV_DELIMITER
                        Set the delimiter of the CSV file
  -H, --hasheader       If set, the header (first line) of the dataset will be
                        omitted
  -I, --allinput        If set, the dataset will be read line by line instead
                        of random samples

```
