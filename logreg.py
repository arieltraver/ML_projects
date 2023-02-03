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


