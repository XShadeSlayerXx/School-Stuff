import pandas as pd
import numpy as np
from sklearn import preprocessing as prep

#alias for function type hint clarity
DataFrame = pd.core.frame.DataFrame
ndarray = np.ndarray



url = 'yacht_hydrodynamics.data'
train_test_split = .8



class Dataset:
    def __init__(self, data_url, percent = .8):
        data_training, data_testing = self.ready_the_data(data_url, percent)
        #assume last row is for the solutions
        number_of_variables = np.size(self.data_training, 1) - 1

    def get_training_data(self) -> ndarray:
        return self.data_training

    def get_testing_data(self) -> ndarray:
        return self.data_testing

    def ready_the_data(self, web_url, training_percent) -> tuple:
        data = self.retrieve_data(web_url = web_url)
        data = self.preprocess_data(data = data)
        return self.split_data(data = data, training_percent = training_percent)

    def preprocess_data(self, data : DataFrame) -> ndarray: # DataFrame? ndarray? normalize converts to numpy?
        ## catagorize data if necessary?
        # prep.OrdinalEncoder()
        ## remove duplicate rows
        ## gaussian distribution?
        # prep.PowerTransformer(method='box-cox', standardize=False)
        # normalize all data between 0 and 1 by row
        normalized = prep.normalize(data)
        return normalized

    def retrieve_data(self, web_url) -> DataFrame:
        return pd.read_csv(web_url, delim_whitespace = True, header = None)

    def split_data(self, data, training_percent : float) -> tuple:
        split_amt = np.ceil(len(data)*training_percent)
        training = data[:split_amt, :]
        testing = data[split_amt:, :]
        return training, testing

    def separate_variables_from_solutions(self, training = True) -> tuple:
        if training:
            nparray = self.data_training.copy()
        else:
            nparray = self.data_testing.copy()
        # return a subsection from 0 to num_variables, and a subsection of only the last column
        return nparray[:, :self.number_of_variables], np[:, self.number_of_variables]

    def get_random_weight_array(self) -> ndarray:
        # return a random array from 0 to 1
        return np.random.rand(self.number_of_variables, 1)

class Regression:
    def __init__(self):
        self.data = Dataset(data_url = url, percent = train_test_split)

        # collect the relevant info from the dataset

        training_data, training_solutions = self.data.separate_variables_from_solutions()
        weights = self.data.get_random_weight_array()
