#!./bin/python
#uses the tutorial here https://medium.com/ml-cheat-sheet/neural-networks-how-to-build-a-cat-classifier-from-scratch-7e8b78c4de2e

import h5py
import numpy as np

#load in the data, default is the cats
def load_dataset(train_images = 'datasets/train_catvnoncat.h5', test_images = 'datasets/test_catvnoncat.h5'):
    train_dataset = h5py.File(train_images, 'r') #create h5 File objects
    test_dataset = h5py.File(test_images, 'r')

    x_train = np.array(train_dataset["train_set_x"][:]) #load the image data into an array
    x_test = np.array(test_dataset["test_set_x"][:])

    y_train = np.array(train_dataset["train_set_y"][:])
    y_test = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:]) #create array for the classes (cat or not)

    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))

    return x_train, y_train, x_test, y_test, classes

def preprocess(x_train, x_test):
    x_train_flat = x_train.reshape(x_train.shape[0], -1).T #unroll data into a 1d vector
    x_test_flat = x_test.reshape(x_test.shape[0], -1).T
    train_set_x = x_train_flat / 255 #divide by the maximum value of a pixel channel
    test_set_x = x_test_flat / 255
    return train_set_x, test_set_x


x_train, y_train, x_test, y_test, classes = load_dataset() #using default
train_set_x, test_set_x = preprocess(x_train, x_test)

def sigmoid(arr):
    """
    sigmoid activation function
    result is between 0 and 1
    args:
    ---arr: scalar or numpy array
    """
    sig = 1 / (1 + np.exp(-arr)) #numpy exp acts on the entire array
    return sig

def initialize_w_zeroes(dim): #for most networks you would init with random values
    zeroes = np.zeros((dim))
    b = 0
    assert(zeroes.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int)) #i don't like python.
    return zeroes, b

def log_cost(A, labelY, m_shape):
    return -1 / m_shape * np.sum(labelY * np.log(A) + (1 - labelY) * np.log(1 - A), axis=1, keepdims = True)


#build a function which computes the cost function AND its gradient.
def propagate(weights, bias, dataX, labelY, cost_fn):
    '''
    returns:
    --- cost
    --- dw: gradient with respepct to weights
    --- db: gradient with respect to bias
    '''
    m_shape = dataX.shape[1] #get the array dimensions
    A = sigmoid(np.dot(weights.T, dataX) + bias) #compute sigmoid of the weights * all the pixels, plus bias
    cost = cost_fn(A, labelY, m_shape)

    dw = 1 / m_shape * np.dot(dataX, (A - labelY).T) #derivative(gradient vector) of sigmoid with respect to weights
    db = 1 / m_shape * np.sum(A - labelY) #derivative(gradient vector) of sigmoid with respect to bias

    return dw, db, cost


def optimize(weights, bias, dataX, dataY, num_iterations, cost_fn, step_size, print_steps = 0):
    '''
    optimizes weights according to what parameters you set.
    ---num_iterations: how many steps should be performed
    ---step_size: how far in the gradient direction do we go with each step
    ---print_steps: if greater than 0, print the cost for every print_steps steps
    '''
    costs = []
    for i in range(num_iterations):
        dw, db, cost = propagate(weights, bias, dataX, dataY, cost_fn)
        weights -= step_size * dw
        bias -= step_size * db
        costs.append(cost)

        if (print_steps > 0 and i % print_steps == 0):
            print(f"step {i}: {cost}")
        
    return weights, bias, dw, db, cost

    
        
        



