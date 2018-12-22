#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'machine-learning-ex2'))
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
path = os.getcwd() + '/ex2/ex2data1.txt'


#%%
data = pd.read_csv(path, header = None, names=['Exam 1 score', 'Exam 2 score', 'Admitted'])
data.describe()


#%%
# visualise raw data

data.plot(kind='scatter', x='Exam 1 score', y='Exam 2 score')


#%%
data['Admitted'].isin([0])


#%%
# assign binary class

# dataframe['column name'] to access that column of data
# dataframe['Admitted'].isin([1]) # returns a series True/False if Admitted column in data(data frame) contains 1
# dataframe[] takes in the series of T/F and shows only True entries
class0 = data[data['Admitted'].isin([0])]
class1 = data[data['Admitted'].isin([1])]


#%%
# visualise raw data graph

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x=class1['Exam 1 score'], y=class1['Exam 2 score'], color='b', marker='o', label='Admitted', s=100)
ax.scatter(x=class0['Exam 1 score'], y=class0['Exam 2 score'], color='r', marker='x', label='Not admitted', s=100)
ax.legend()
ax.set_xlabel('Exam 1 score')
ax.set_ylabel('Exam 2 score')

#%% [markdown]
# using traditional GDA

#%%
# define sigmoid function using lambda function

sigmoid = lambda z: 1 / (1 + np.exp(-z))


#%%
# test sigmoid function is working

test_range = np.arange(-10, 10, 0.1)

plt.figure(figsize=(12,8))
plt.scatter(x=test_range, y=sigmoid(test_range))


#%%
# create design matrix

data.insert(0, 'x0', 1) # insert feature 0
X = data.iloc[:,:-1]
y = data.iloc[:,-1:]
theta = np.mat(np.zeros((X.shape[1],1)))

X, y = np.mat(X.values), np.mat(y.values)
X.shape, theta.shape, y.shape


#%%
# define cost function for logistics regression

def cost_function(theta, X, y):
    predicted_y = sigmoid(X * theta)
    return np.sum(1 / X.shape[0] * (-y.T * np.log(predicted_y) - (1 - y).T * np.log(1 - predicted_y)))


#%%
# confirm cost_function works

cost_function(theta, X, y)


#%%
# implement normal gradient descent

def gradient_descent(X, y, theta, alpha, iters):
    cost = np.zeros(iters)

    for i in range(iters):
        theta = theta - (alpha / X.shape[0] * X.T * (sigmoid(X * theta) - y))
        cost[i] = cost_function(theta, X, y)
    return theta, cost


#%%
# plot graph for iters vs cost

def plot_cost(cost, iters):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Training Epoch')
    ax.set_xlim(0, iters)
    ax.set_ylim(cost[iters-1], cost[0])
    ax.grid(True)


#%%
# runs normal GDA and plots graph

alpha = 0.001
iters = 1000
theta = np.mat(np.zeros((X.shape[1],1)))

theta_best, cost = gradient_descent(X, y, theta, alpha, iters)
print(theta_best)
print(np.amin(cost))
plot_cost(cost, iters)

#%% [markdown]
# using opt.fmin_tnc

#%%
# define cost function for logistics regression with opt.fmin_tnc

def cost_function(theta, X, y):
    # theta gets auto converted to array:
#     /usr/local/lib/python3.6/site-packages/numpy/matrixlib/defmatrix.py in __mul__(self, other)
#         307         if isinstance(other, (N.ndarray, list, tuple)) :
#         308             # This promotes 1-D vectors to row vectors
# has to convert theta from n-D array to 1 x n matrix, then transpose to n x 1 matrix
    theta = np.mat(theta).T

    predicted_y = sigmoid(X * theta)
    return np.sum(1 / X.shape[0] * (-y.T * np.log(predicted_y) - (1 - y).T * np.log(1 - predicted_y)))


#%%
# implement gradient function, partial derivative of cost function
# using lambda function with vector
# has to convert theta from n-D array to 1 x n matrix, then reshape to n x 1 matrix

gradient = lambda theta, X, y: np.array((1 / X.shape[0] * X.T * (sigmoid(X * np.mat(theta).reshape((-1,1))) - y)).T).reshape((X.shape[1]))


#%%
# test gradient function is working

gradient(theta, X, y)
theta.shape, X.shape, y.shape


#%%
# uses scipy's truncated newton implementation to find optimal parameters

import scipy.optimize as opt
result = opt.fmin_tnc(func=cost_function, x0=theta, fprime=gradient, args=(X, y))
theta_best = result[0]
print(theta_best)
cost_function(theta_best, X, y)


#%%
def predict(theta, X):
    theta = np.mat(theta).T
    probability = sigmoid(X * theta)
    return [1 if x >= .5 else 0 for x in probability] # return as array of 0 or 1


#%%
predicted = predict(theta_best, X)


#%%
# add predicted column into dataframe, catches duplicate error
try:
    data.insert(4, 'predicted', predicted)
except ValueError:
    pass
data.head(10)


#%%
predicted_class0 = data[data['predicted'].isin([0])]
predicted_class1 = data[data['predicted'].isin([1])]


#%%
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x=class1['Exam 1 score'], y=class1['Exam 2 score'], color='b', marker='o', label='Admitted', s=100)
ax.scatter(x=class0['Exam 1 score'], y=class0['Exam 2 score'], color='r', marker='x', label='Not admitted', s=100)

ax.scatter(x=predicted_class1['Exam 1 score'], y=predicted_class1['Exam 2 score'], color='k', marker='o', label='Predicted Admitted', s=150)
ax.scatter(x=predicted_class0['Exam 1 score'], y=predicted_class0['Exam 2 score'], color='grey', marker='x', label='Predicted Not admitted', s=150)

ax.legend()
ax.set_xlabel('Exam 1 score')
ax.set_ylabel('Exam 2 score')


#%%
# test accuracy of predicted against y

# list comprehension
# need to convert y (matrix) to array, then ravel, then list, as predicted is list
correct = [1 if a == b else 0 for (a, b) in zip(list(np.array(y).ravel()), predicted)]
accurary = np.sum(correct) / len(correct)
print(accurary)


#%%
# plot decision boundary
# manually generate grid and calculate score for every grid

step = 100
exam_grid = np.hstack((
    np.linspace(30, 100, step).reshape((-1,1)),
    np.linspace(30, 100, step).reshape((-1,1))
))

predicted_grid = np.zeros((step, step))

# calculates each y by looping through every combination of grid
for i in range(step):
    for j in range(step):
        predicted_grid[i][j] = sum(predict(theta_best, np.array((1.0, exam_grid[i][0], exam_grid[j][1])))) # use sum() to convert array to int

plt.figure(figsize=((12,8)))
plt.contourf(exam_grid[:,0], exam_grid[:,1], predicted_grid, 50, cmap = 'RdGy')


#%%
# plot decision boundary using mesh

x_min, x_max = 30, 100
y_min, y_max = 30, 100
step = 100

# create a grid mesh
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, step),
    np.linspace(y_min, y_max, step)
)

# ravel the grid to a list, then stack together
exam_grid = np.hstack((np.ones(xx.ravel().shape[0]).reshape(-1,1), xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1)))

predicted_grid = predict(theta_best, exam_grid) # get a list of 0 and 1
predicted_grid = np.array(predicted_grid).reshape(xx.shape) # convert to np array, then reshape to x by x grid, for contour plot later
# print(predicted_grid)

plt.figure(figsize=((12,8)))
plt.contourf(xx, yy, predicted_grid, cmap = 'RdGy')


