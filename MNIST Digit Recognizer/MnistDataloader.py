#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np # linear algebra
import pandas as pd

data = pd.read_csv('/Users/nicomeiswinkel/Python/Neural_Network/MNIST/train.csv')
data_test = pd.read_csv('/Users/nicomeiswinkel/Python/Neural_Network/MNIST/test.csv')
data = np.array(data)
data_test = np.array(data_test)
m, n = data.shape()
np.random.shuffle(data)

data_dev = data[0:1000].T
X_dev = data_dev[0]
Y_dev = data_dev[1:n]


