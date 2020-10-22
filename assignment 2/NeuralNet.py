#####################################################################################################################
#   Assignment 2: Neural Network Programming
#   This is a starter code in Python 3.6 for a 1-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   NeuralNet class init method takes file path as parameter and splits it into train and test part
#         - it assumes that the last column will the label (output) column
#   h - number of neurons in the hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   W_hidden - weight matrix connecting input to hidden layer
#   Wb_hidden - bias matrix for the hidden layer
#   W_output - weight matrix connecting hidden layer to output layer
#   Wb_output - bias matrix connecting hidden layer to output layer
#   deltaOut - delta for output unit (see slides for definition)
#   deltaHidden - delta for hidden unit (see slides for definition)
#   other symbols have self-explanatory meaning
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


#globals bc easier to see, could just as easily go down to the __main__ function

#only sigmoid works and im not entirelu sure why
activation_function = 'relu'
#print the weight vectors?
print_header = False

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv'

number_of_neurons = 4
learn_rate = .25
max_iters = 60000


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as prep
from os.path import exists as file_exists


class NeuralNet:
    def __init__(self, dataFile, h=4, test_data = None):
        #np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h represents the number of neurons in the hidden layer
        raw_input = pd.read_csv(dataFile)
        processed_data = self.preprocess(raw_input)
        if test_data is None:
            self.train_dataset, self.test_dataset = train_test_split(processed_data)
        else:
            self.train_dataset = processed_data
            raw_input = pd.read_csv(test_data)
            self.test_dataset = self.preprocess(raw_input)
        ncols = len(self.train_dataset.columns)
        nrows = len(self.train_dataset.index)
        nrows_test = len(self.test_dataset.index)
        self.X = self.train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.X_test = self.test_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows_test, ncols-1)
        self.y = self.train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        self.y_test = self.test_dataset.iloc[:, (ncols-1)].values.reshape(nrows_test, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[1])
        if not isinstance(self.y[0], np.ndarray):
            self.output_layer_size = 1
        else:
            self.output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.W_hidden = 2 * np.random.random((input_layer_size, h)) - 1
        self.Wb_hidden = 2 * np.random.random((1, h)) - 1

        self.W_output = 2 * np.random.random((h, self.output_layer_size)) - 1
        self.Wb_output = np.ones((1, self.output_layer_size))

        self.deltaOut = np.zeros((self.output_layer_size, 1))
        self.deltaHidden = np.zeros((h, 1))
        self.h = h

    def preprocess(self, data):
        le = prep.LabelEncoder()
        for col in data.columns:
            #if the column isn't a number...
            if not np.issubdtype(data[col].dtype, np.number):
                data[col] = data[col].fillna(0)
                data[col] = le.fit_transform(data[col])

        # normalize all data between 0 and 1 by row
        normalized = prep.normalize(data)
        bias = np.ones([np.size(normalized,0),1])
        normalized = np.append(bias, normalized, axis = 1)
        #randomize rows
        np.random.shuffle(normalized)
        return pd.DataFrame(normalized)

    def __activation(self, x, activation="sigmoid"):
        try:
            activation = ('sigmoid', 'tanh', 'relu')[int(activation)]
        except:
            activation = activation.lower() #idk why it would be uppercase but /shrug
        if activation == "sigmoid":
            return self.__sigmoid(x)
        elif activation == "tanh":
            return self.__tanh(x)
        elif activation == "relu":
            return self.__relu(x)

    def __activation_derivative(self, x, activation="sigmoid"):
        try:
            activation = ('sigmoid', 'tanh', 'relu')[int(activation)]
        except:
            activation = activation.lower() #idk why it would be uppercase but /shrug
        if activation == "sigmoid":
            return self.__sigmoid_derivative(x)
        elif activation == "tanh":
            return self.__tanh_derivative(x)
        elif activation == "relu":
            return self.__relu_derivative(x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh(self, x):
        return np.tanh(x)
        # return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def __tanh_derivative(self, x):
        return 1 - np.power(x,2)
        # return 1 - (self.__tanh(x)) ** 2

    def __relu(self, x):
        return x * (x > 0)
        # tempx = []
        # for i, row in enumerate(x):
        #     tempx.append([])
        #     for val in row:
        #         tempx[i].append(max(0,val))
        # return np.array(tempx)

    def __relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
        # tempx = []
        # for i, row in enumerate(x):
        #     tempx.append([])
        #     for val in row:
        #         tempx[i].append(1 if val > 0 else 0)
        # return np.array(tempx)

    def get_error(self, test = False, which_activation = 'sigmoid'):
        if not test:
            which_y = self.y
        else:
            which_y = self.y_test
        out = self.forward_pass(activation = which_activation, test = test)
        error = 0.5 * np.power((out - which_y), 2)
        return out, error

    # Below is the training function

    def train(self, max_iterations=60000, learning_rate=0.25, which_activation = 'sigmoid', header = True):
        for iteration in range(max_iterations):
            out, error = self.get_error(test = False, which_activation = which_activation)
            self.backward_pass(out, activation = which_activation)

            update_weight_output = learning_rate * np.dot(self.X_hidden.T, self.deltaOut)
            update_weight_output_b = learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaOut)

            update_weight_hidden = learning_rate * np.dot(self.X.T, self.deltaHidden)
            update_weight_hidden_b = learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaHidden)

            self.W_output += update_weight_output
            self.Wb_output += update_weight_output_b
            self.W_hidden += update_weight_hidden
            self.Wb_hidden += update_weight_hidden_b

        if header:
            print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
            print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_hidden))
            print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_output))

            print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_hidden))
            print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_output))

    def forward_pass(self, activation="sigmoid", test = False):
        # pass our inputs through our neural network
        if test:
            which_x = self.X_test
        else:
            which_x = self.X
        in_hidden = np.dot(which_x, self.W_hidden) + self.Wb_hidden
        self.X_hidden = self.__activation(in_hidden, activation)
        in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output
        out = self.__activation(in_output, activation)
        return out

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_delta(activation)

    def compute_output_delta(self, out, activation="sigmoid"):
        delta_output = (self.y - out) * (self.__activation_derivative(out, activation))

        self.deltaOut = delta_output

    def compute_hidden_delta(self, activation="sigmoid"):
        delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__activation_derivative(self.X_hidden, activation))

        self.deltaHidden = delta_hidden_layer

    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, test = True, which_activation = 'sigmoid'):
        _, error = self.get_error(test, which_activation)
        return sum(error)


if __name__ == "__main__":
    print_tests = False
    # perform pre-processing of both training and test part of the test_dataset
    # split into train and test parts if needed
    kwargs = {'dataFile': 'train.csv' if file_exists('train.csv') else url,
            'h': number_of_neurons,
            'test_data': 'test.csv' if file_exists('test.csv') else None}
    kwargs_train = {'max_iterations':max_iters,
                    'learning_rate':learn_rate,
                    'header': print_header,
                    'which_activation': activation_function}
    if print_tests:
        output = {}
        neurons_params = [1,2,4,8]
        activation_params = ['tanh', 'relu', 'sigmoid']
        learn_rate_params = [.1,.25,.5]
        max_iters_param = [1000,10000,100000]
        #create a 2d array for output, for every neuron
        for num in activation_params:
            #learn rate is on the y axis, max iterations is on the x axis
            col = len(learn_rate_params)
            row = len(max_iters_param)
            output[num] = [[0 for i in range(col+1)] for j in range(row+1)]

            for i, learn in enumerate(learn_rate_params):
                output[num][i+1][0] = learn
            for j, iters in enumerate(max_iters_param):
                output[num][0][j+1] = iters

        for activation in activation_params:
            kwargs_train['which_activation'] = activation
            for l, learn in enumerate(learn_rate_params):
                kwargs_train['learning_rate'] = learn
                for j, max_it in enumerate(max_iters_param):
                    kwargs_train['max_iterations'] = max_it
                    neural_network = NeuralNet(**kwargs)
                    neural_network.train(**kwargs_train)
                    #trainError = neural_network.predict(test = False, which_activation = activation_function)
                    testError = neural_network.predict(test = True, which_activation = activation_function)
                    # print('# Neurons :',kwargs['h'])
                    # print('Max Iterations :',kwargs_train['max_iterations'])
                    # print('Learning Rate :',kwargs_train['learning_rate'])
                    output[activation][l+1][j+1] = str(testError[0])
                    print(activation, l, j)
                    # print("Train error = " + str(trainError[0]))
                    print("Test error = " + str(testError[0]))
        output_string = ''
        for neuron, array in output.items():
            output_string += f'Activation Function: {neuron}\n'
            for row in array:
                output_string += ' | '.join(f'{str(x):13.13}' for x in row) + '\n'
            output_string += '\n'
        print(output_string)
    else:
        neural_network = NeuralNet(**kwargs)
        neural_network.train(**kwargs_train)
        trainError = neural_network.predict(test = False, which_activation = activation_function)
        testError = neural_network.predict(test = True, which_activation = activation_function)
        print('# Neurons :',kwargs['h'])
        print('Max Iterations :',kwargs_train['max_iterations'])
        print('Learning Rate :',kwargs_train['learning_rate'])
        print("Train error = " + str(trainError))
        print("Test error = " + str(testError))