import pandas as pd
import numpy as np
from sklearn import preprocessing as prep

#alias for function type hint clarity
DataFrame = pd.core.frame.DataFrame
ndarray = np.ndarray

class Dataset:
    def __init__(self, data_url, percent = .8, *, header = None):
        self.data_training, self.data_testing = self.ready_the_data(data_url, percent, header = header)
        #assume last row is for the solutions
        self.number_of_variables = np.size(self.data_training, 1) - 1

    def get_training_data(self) -> ndarray:
        return self.data_training

    def get_testing_data(self) -> ndarray:
        return self.data_testing

    def ready_the_data(self, web_url, training_percent,*, header = None) -> tuple:
        data = self.retrieve_data(web_url = web_url, header = header)
        data = self.preprocess_data(data = data)
        return self.split_data(data = data, training_percent = training_percent)

    def preprocess_data(self, data : DataFrame) -> ndarray: # DataFrame? ndarray? normalize converts to numpy?
        ## catagorize data if necessary?
        le = prep.LabelEncoder()
        for col in data.columns:
            #if the column isn't a number...
            if not np.issubdtype(data[col].dtype, np.number):
                data[col] = data[col].fillna('None')
                data[col] = le.fit_transform(data[col])

        # normalize all data between 0 and 1 by row
        normalized = prep.normalize(data)
        bias = np.ones([np.size(normalized,0),1])
        normalized = np.append(bias, normalized, axis = 1)
        #randomize rows
        np.random.shuffle(normalized)
        return normalized

    def retrieve_data(self, web_url, *, header = None) -> DataFrame:
        if header == True:
            header = 'infer'
        return pd.read_csv(web_url, delim_whitespace = True, header = header)

    def split_data(self, data, training_percent : float) -> tuple:
        split_amt = int(np.ceil(len(data)*training_percent))
        training = data[:split_amt, :]
        testing = data[split_amt:, :]
        return training, testing

    def separate_variables_from_solutions(self, training = True) -> tuple:
        if training:
            nparray = self.data_training.copy()
        else:
            nparray = self.data_testing.copy()
        # return a subsection from 0 to num_variables, and a subsection of only the last column
        return nparray[:, :self.number_of_variables], nparray[:, self.number_of_variables]

    def get_random_weight_array(self) -> ndarray:
        # return a random array from 0 to 1
        return np.random.rand(self.number_of_variables, 1)

class Regression:
    def __init__(self, url, train_test_split = .8,
                 number_of_iterations : int = 1500, learning_rate : float = .1
                 , *, header = None):
        #initialize the data
        self.dataset = Dataset(data_url = url, percent = train_test_split, header = header)

        # collect the relevant info from the dataset
        self.data, self.solutions = self.initialize_data()
        self.test_data, self.test_solutions = self.initialize_data(training_data = False)

        self.num_iterations = number_of_iterations

        self.learning_rate = learning_rate
        self.weights = self.dataset.get_random_weight_array()

    def set_iterations(self, num : int):
        self.num_iterations = num

    def get_numrows(self, training_data = True):
        if training_data:
            return self.data.shape[0]
        else:
            return self.test_data.shape[0]

    def get_numcols(self):
        return self.data.shape[1]

    def which_sols(self, training_data = True):
        if training_data:
            return self.solutions
        else:
            return self.test_solutions

    def initialize_data(self, training_data = True):
        return self.dataset.separate_variables_from_solutions(training = training_data)

    def calculate_new_weights(self):
        # new weight is the old weight - learning_rate * (mean of (xi - error))
        separate_vars = np.empty([self.get_numcols(),self.get_numrows()])
        error = self.calculate_error()
        for i in range(self.get_numcols()):
            separate_vars[i] = self.data[:,i] * error[0]
            self.weights[i] -= self.learning_rate * np.mean(separate_vars[i])

    def calculate_theoretical_solution(self, training_data = True):
        # dot product of the arrays and the weight array
        if training_data:
            dt = self.data
        else:
            dt = self.test_data
        return dt.dot(self.weights)

    def calculate_error(self, training_data = True):
        # theoretical - solution
        return self.calculate_theoretical_solution(training_data).T - self.which_sols(training_data)

    def calculate_mse(self, training_data = True):
        # sum of all error's squared divided by 2N
        error = self.calculate_error(training_data)
        return np.sum(error**2)/(2*self.get_numrows(training_data))

    def print_logs(self, where = None, header = None):
        output = self.output_info()
        if header is not None:
            output = header + '\n' + output
        if where is not None:
            with open(where, 'a') as file:
                file.write(output)
        else:
            print(output)

    def start_learning(self, number_data_points = 50) -> tuple:
        number_data_points -= 1 # to space them out from 0 to max equally
        output = []
        for x in range(self.num_iterations+1):
            if x % int(np.round(self.num_iterations/number_data_points)) == 0:
                output.append(self.calculate_mse())
            self.calculate_new_weights()
        output.append(self.calculate_mse())
        step_amt = int(np.round(self.num_iterations/number_data_points))
        steps = [x*step_amt for x in range(len(output))]
        return steps, output

    def output_info(self) -> str:
        output = f'Weights:\n{self.weights}\n'
        output += f'Learning Rate: {self.learning_rate}\n'
        output += f'Number of iterations: {self.num_iterations}\n'
        mse = self.calculate_mse()
        output += f'MSE: {mse}\n'
        #output += f'Accuracy v Training: {1 - mse}\n'
        test_mse = self.calculate_mse(training_data = False)
        output += f'MSE v Testing: {test_mse}\n'
        #output += f'Accuracy v Testing: {1 - test_mse}\n'
        return output

    def __iter__(self):
        self.iterations = self.num_iterations
        return self

    def __next__(self):
        if self.iterations > 0:
            self.iterations -= 1
            #step through the algorithm again
            self.calculate_new_weights()
            #print(f'Iteration: {self.num_iterations - self.iterations} -- MSE: {self.calculate_mse()}')
            return self.calculate_mse()
        else:
            raise StopIteration

    def __repr__(self):
        return self.output_info()

