from LinearRegression import Regression as rg
import matplotlib.pyplot as plt

#multiple output variables is possible but not currently supported

#where the data is stored
# 2 examples, one is a better overview, the other is proof this works with multiple datasets
url1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
url2 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data'

# change this to the dataset you want
url = url1

#percentage of data that goes to training vs testing
train_test_split = .8

#file name to the file or 'None' to console
logFile = None
#display graph after training?
display_graph = True

##optional parameters:
#header is either None or the row names in a list e.g. ['city', 'state', 'etc']
#'header = True' if header is included in the file
header = None
#about 500 was found to be the drop off point
number_of_iterations = 1500000
number_of_data_points = 50
#about .1 was found to be optimal for the yacht dataset
learning_rate = .01

regression_example = rg(url, train_test_split, number_of_iterations, learning_rate, header = header)

regression_example.print_logs(logFile, 'My algorithm:\n\nInitial')

x, y = regression_example.start_learning(number_of_data_points)

regression_example.print_logs(logFile, 'Final')

if display_graph:
    plt.title('Iterations vs MSE')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.scatter(x,y)
    plt.show()
