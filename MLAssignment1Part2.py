from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'

df = pd.read_csv(url, header = None, delim_whitespace = True)

df.fillna(method = 'ffill', inplace = True)

X = df.iloc[:,:-1].values
y = df.iloc[:,1].values

X_train, X_test, y_train, y_test = tts(X, y, test_size = .2, random_state = 0)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("MSE: ",metrics.mean_squared_error(y_test, y_pred))