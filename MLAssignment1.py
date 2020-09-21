from LinearRegression import Regression as rg
import matplotlib.pyplot as plt

# multiple output variables is possible but not currently supported

# 'file name' to the file or 'None' to console
# for example 'iteration_150000_learnrate_1_tenth.txt'
logFile = None
# same as logFile, but .png instead of .txt
logGraphFile = None

# about 500 was found to be a drop off point
# but 15000 still reduced the mse further
number_of_iterations = 1500
number_of_data_points = 50
#about .1 was found to be optimal for the yacht dataset
learning_rate = .1

# where the data is stored
# 2 examples, one is a better overview, the other is proof this works with multiple datasets
url1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
url2 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data'
url = url1

# percentage of data that goes to training vs testing
# .8 means 80% training, 20% testing
train_test_split = .8

#display graph after training?
display_graph = False

## optional parameters:
# header is either None or the row names in a list e.g. ['city', 'state', 'etc']
# 'header = True' if header is included in the file
header = None

regression_example = rg(url, train_test_split, number_of_iterations, learning_rate, header = header)

regression_example.print_logs(logFile, 'Initial')

x, y = regression_example.start_learning(number_of_data_points)

regression_example.print_logs(logFile, '\n\nFinal')

plt.title('Iterations vs MSE')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.scatter(x,y)

if logGraphFile is not None:
    plt.savefig(logGraphFile)

if display_graph:
    plt.show()