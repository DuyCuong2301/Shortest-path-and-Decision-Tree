import numpy as np
import math


def Gini_index(y):
    
    unique_label = np.unique(y)
    square_p = 0.0

    for label in unique_label:
        count = len(y[y==label])
        p = count / len(y)
        square_p += p * p

    gini = 1 - square_p 
    return gini

def Entropy(y):
  
    unique_label = np.unique(y)
    entropy = 0.0
    log2 = lambda x: math.log(x) / math.log(2)

    for label in unique_label:
        count = len(y[y==label])
        p = count / len(y)
        entropy += - p * log2(p)
    
    return entropy


def Mean_Square_Error(y):
    
    y_pred = np.mean(y)
    mse = np.mean(np.power(y - y_pred, 2))
    return mse

def Mean_absolute_error(y):
    
    mae = np.mean(np.abs((y - np.median(y))))
    return mae

def divide_on_feature(X, feature_i, threshold):
    """ Divide dataset based on if sample value on feature index is larger than
        the given threshold"""
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold
    
    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return X_1, X_2
