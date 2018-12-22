
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
path = '/Volumes/Cloud/Dropbox/Files/Code/Machine Learning/machine-learning-ex1/ex1/ex1data1.txt'


#%%
# load data (csv) into a data frame, an object
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head() # view first n entries, default 5
# data.tail()


#%%
data.describe() # calculates stats


#%%
# plot data, wrapper for matplotlib
data.plot(kind='scatter', x='Population', y='Profit', xlim=(0,25), ylim=(-5,30),figsize=(12,8),grid=True)

#%% [markdown]
# implement linear regression model on the data

#%%
# define squared error cost function
def compute_cost(X, y, theta):
    predicted_y = X * theta # compute predicted y using linear algebra
    error = predicted_y - y
#     squared_error = np.power(error, 2)
    squared_error = error.T * error
#     sum_squared_error = np.sum(squared_error)
    squared_error_cost = squared_error / (2 * X.shape[0])
    return np.sum(squared_error_cost) # convert to raw value instead of matrix


#%%
# manipulate data from pandas data frame into X, y and theta
# store all data as matrix of m x n
# not array, tho array is easier to manipulate using multiplication

data.insert(0, 'x0', 1) # insert feature 0

# slice the data to get design matrix
X = data.iloc[:,:-1] # set X to be the first to second last columns
y = data.iloc[:,-1:] # set y to be the last column
theta = np.mat(np.zeros((X.shape[1], 1)))
# initialise empty theta, using np.array for decimal points
# if used np.array[0,0], then cells are initialised  as int
# alt method use np.zeros
# uses (X.shape[1],1) to auto assign the dimension of theta
# results in 2*1 matrix

X, y = np.mat(X.values), np.mat(y.values) # convert to matrix


#%%
# test compute_cost function and variables are initialised properly
# answer should be 32.072733877455676
compute_cost(X, y, theta)


#%%
# # define gradient desecnt function (loop)
# def gradient_descent(X, y, theta, alpha, iters):
#     cost = np.zeros(iters) # create an array for storing history of costs after each iteration
#     theta_number = int(theta.shape[0]) # number of theta
#     temp = theta # initialise temp, for storing calculated theta

#     for i in range(iters):
#         error = (X * theta) - y # returns a list of all errors
#         print(error)

#         for j in range(theta_number):
# #             gradient = np.sum(np.multiply(error, X[:, j])) # multiply returns scalar product of each row, then sum up
# #             <class 'numpy.matrixlib.defmatrix.matrix'> <class 'numpy.matrixlib.defmatrix.matrix'>
#             gradient = (error.T *  X[:, j]) # returns the dot product directly (matrix([number]), without sum (converts to float/int)
#             increment = alpha / X.shape[0] * gradient
#             temp[j, 0] = theta[j, 0] - increment

#         theta = temp
#         cost[i] = compute_cost(X, y, theta)

#     return theta, cost


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
iters = 10000
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
# returns the data in array
x = np.array(X[:, 1].A1)
y_predicted = theta_best[0,0] + theta_best[1,0]*x # calculate predicted y from theta


#%%
# vectorised version
# returns the data in matrix
# does not affect plotting
# y_predicted = X * theta_best


#%%
fig, ax = plt.subplots(figsize=(12,8)) # create fig and axes objects
ax.plot(x, y_predicted, 'r', label='Prediction') # plot x, y and 'r' red line, assign label, create a straight line
ax.scatter(data.Population, data.Profit, label='Training Data') # overlay training data as scatter plot
ax.legend(loc=0)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
ax.set_xlim(0,25)
ax.set_ylim(-5,30)
ax.grid(True)


#%%
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs Training Epoch')
ax.set_xlim(0, iters)
ax.set_ylim(cost[iters-1], cost[0])
ax.grid(True)

#%% [markdown]
# Normal equation method

#%%
from scipy import linalg
theta_best = np.linalg.pinv(X.T*X)*X.T*y
theta_best


#%%
compute_cost(X, y, theta_best)

#%% [markdown]
# Using scikit-learn

#%%
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)


#%%
y_predicted2 = model.predict(X).flatten()
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, y_predicted2, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.scatter(x, y_predicted, label='Predicted data using np.linspace, then calculate manually')
ax.scatter(x, y_predicted2, label='Predicted data using sk predict')
ax.legend(loc=2)
ax.set_title('Predicted Profit vs. Population Size')
ax.set_xlabel('Population')
ax.set_ylabel('Profit')

#%% [markdown]
# Plotting J(theta) vs theta curve

#%%
# old method, manually creating temp then copy to theta_set

# iters = 1000
# temp = np.array(np.linspace(0, 1, iters)) # create a temp array of random data of theta1
# theta_set = np.zeros((temp.shape[0], 1)) # create a 1000 x 1 array of 0.0
# theta_set[:,0] = temp # copy the random data to the first column of theta_set
# theta_set = np.insert(theta_set, 0, theta_best[0], axis=1) # insert a whole column of number, (array, index, number, axis for whole column)


#%%
# plot J(theta) vs theta curve
# set theta0 to the value calculated above from theta_best
# results in simple 1D curve

iters = 1000
theta_set = np.array(np.linspace(0, 2.4, iters)).reshape((-1,1)) # create array of random data of theta1, then reshape to 1000*1 (-1 for as many as possible), as .T does not work
theta_set = np.insert(theta_set, 0, theta_best[0], axis=1) # insert a whole column of number, (array, index, number, axis for whole column)
cost = np.zeros(iters)
for i in range(iters):
    theta = theta_set[i].reshape((2,1)) # convert 1*2 array to 2*1 using reshape, .T does not work
    cost[i] = compute_cost(X, y, theta)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(theta_set[:,1], cost, label='set θ0 = -3.89578088') # plot cost vs theta1
ax.set_xlabel('θ1')
ax.set_ylabel('J(θ1)')
ax.set_title('J(θ1) vs θ1')
ax.grid(True)
ax.legend(loc=0)


#%%
# plot J(theta) vs theta curve
# set theta1 to the value calculated above from theta_best
# results in simple 1D curve

iters = 1000
theta_set = np.array(np.linspace(-10, 2.5, iters)).reshape((-1,1)) # create array of random data of theta1, then reshape to 1000*1 (-1 for as many as possible), as .T does not work
theta_set = np.insert(theta_set, 1, theta_best[1], axis=1) # insert a whole column of number, (array, index, number, axis for whole column)
cost = np.zeros(iters)
for i in range(iters):
    theta = theta_set[i].reshape((2,1)) # convert 1*2 array to 2*1 using reshape, .T does not work
    cost[i] = compute_cost(X, y, theta)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(theta_set[:,0], cost, label='set θ1') # plot cost vs theta1
ax.legend(loc=0)
ax.set_title('J(θ1) vs θ1')
ax.set_xlabel('θ1')
ax.set_ylabel('J(θ1)')
ax.grid(True)


#%%
# plot J(theta) vs theta 2D contour

step = 100
# create 2 arrays of random data of theta0 and 1, then reshape to -1*1
# use np.hstack to join the 2 arrays together, expanding column
theta_set = np.hstack((
	np.linspace(-10,2.5, step).reshape((-1,1)),
	np.linspace(0, 2.4, step).reshape((-1,1))
))
cost = np.zeros((step,step)) # create 1000x1000 array for cost, for storing the cost for each pair of theta0, theta1
cost
for i in range(step):
    for j in range(step):
        # create a theta array for computation later
        # 2 values of theta
        theta = np.array([[theta_set[i][0]], [theta_set[j][1]]])
        cost[i][j] = compute_cost(X, y, theta)

fig = plt.contourf(theta_set[:,0], theta_set[:,1], cost, 50, cmap = 'RdGy') # create a filled contour, using colormap Red-Grey
plt.colorbar() # create colorbar for scale
