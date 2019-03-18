# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    
    

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    # x: row=instance :: column=classes=nodes
    x_exp = np.exp(x)
    sum_exp = np.sum(x_exp, axis=1)
    sum_exp = np.array([sum_exp]) # row=1 :: column=sum for each instance
    probabilities = x_exp / sum_exp.T
    return probabilities


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    batch_size = len(y)
    cost = (np.log(y) * t) + (np.log(1-y) * (1-t))
    cost *= -1/batch_size
    cost = np.sum(cost)
    return cost 


def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)
