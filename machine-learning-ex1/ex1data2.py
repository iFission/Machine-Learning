#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'machine-learning-ex1'))
	print(os.getcwd())
except:
	pass

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
import os


#%%
path = os.getcwd() + '/ex1/ex1data2.txt'


#%%
# load data (csv) into a data frame, an object
data = pd.read_csv(path, header=None, names=['Size', 'Bedroom', 'Price'])
data.head() # view first n entries, default 5
# data.tail()


#%%
data.describe() # calculates stats


#%%
data = (data-data.mean()) / data.std()
data.head()

#%% [markdown]
# implement linear regression model on the data
# does not have to change anything from ex1data1, as both algorithms uses linear algebra, able to handle data of any dimensions

#%%
# define squared error cost function
def compute_cost(X, y, theta):
    predicted_y = X * theta # compute predicted y using linear algebra
    error = predicted_y - y
    squared_error = np.power(error, 2)
    sum_squared_error = np.sum(squared_error)
    squared_error_cost = sum_squared_error / (2 * X.shape[0])
    return squared_error_cost


#%%
# manipulate data from pandas data frame into X, y and theta
data.insert(0, 'x0', 1) # insert feature 0


#%%
# slice the data
X = data.iloc[:,:-1] # set X to be the first to second last columns
y = data.iloc[:,-1:] # set y to be the last column
theta = np.mat(np.zeros(X.shape[1])).T
# initialise empty theta, using np.array for decimal points
# if used np.array[0,0], then cells are initialised  as int
# alt method use np.zeros
# uses X.shape[1] to auto assign the dimension of theta
# results in 2*1 matrix

X, y = np.mat(X.values), np.mat(y.values) # convert to matrix


#%%
# test compute_cost function and variables are initialised properly
# answer should be 32.072733877455676
compute_cost(X, y, theta)


#%%
# define gradient desecnt function (vectorized version of above)
def gradient_descent(X, y, theta, alpha, iters):
    cost = np.zeros(iters) # create an array for storing history of costs after each iteration

    for i in range(iters):
        # error = (X * theta) - y
        # gradient = X.T * error # transpose X features, to solve for matrix vector product, X is 100 * n, error is 100*1
        # increment = alpha / X.shape[0] * gradient
        # theta = theta-increment
        theta = theta - alpha / X.shape[0] * (X.T * ((X * theta) - y))
        cost[i] = compute_cost(X, y, theta)
    return theta, cost


#%%
# initialise variables for learning rate and itertations
alpha = 0.01
iters = 1000
theta = np.mat(np.zeros((X.shape[1], 1)))

# perform gradient descent
theta_best, cost = gradient_descent(X, y, theta, alpha, iters)
theta_best


#%%
compute_cost(X, y, theta_best)

#%% [markdown]
# Plotting graphs

#%%
# x = np.linspace(data.Population.min(), data.Population.max(), 100) # create an array of evenly space values, used as x
# y_predicted = theta_best[0,0] + theta_best[1,0]*x # calculate predicted y from theta
# fig, ax = plt.subplots(figsize=(12,8)) # create fig and axes objects
# ax.plot(x, y_predicted, 'r', label='Prediction') # plot x, y and 'r' red line, assign label
# ax.scatter(data.Population, data.Profit, label='Training Data') # overlay training data as scatter plot
# ax.legend(loc=0)
# ax.set_xlabel('Population')
# ax.set_ylabel('Profit')
# ax.set_title('Predicted Profit vs. Population Size')


#%%
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs Training Epoch')
ax.grid(True)
ax.set_xlim(0, iters)
ax.set_ylim(cost[iters-1], cost[0])

#%% [markdown]
# Normal equation method

#%%
from scipy import linalg
theta_best = np.linalg.pinv(X.T*X)*X.T*y
theta_best


#%%
compute_cost(X, y, theta_best)


