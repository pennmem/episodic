"""
array_operations.py

Author:
    Daniel Schonhaut
    Computational Memory Lab
    University of Pennsylvania
    daniel.schonhaut@gmail.com

Description: 
    Functions for indexing and operating on arrays.

Last Edited: 
    7/9/20
"""
# import mkl
# mkl.set_num_threads(1)
import numpy as np
import pandas as pd

def cos_sim(x, y):
    """Return the cosine similarity between x and y."""
    x = np.array(x)
    y = np.array(y)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    
def dice(x, y):
    """Return the Dice coefficient between x and y.
    
    DSC = (2 * sum(X&Y)) / (sum(X) + sum(Y))
    """
    x = np.array(x)
    y = np.array(y)
    return (2 * np.isin(np.nonzero(x), np.nonzero(y)).sum()) / (np.count_nonzero(x) + np.count_nonzero(y))
       
def make_epochs(vec, epoch_inds, epoch_size):
    """Epoch vec around a series of event indices.
    
    Parameters
    ----------
    vec : np.ndarray
        A vector of data that the epochs are drawn from.
    epoch_inds : np.ndarray
        Indices of vec that epochs are centered around.
    epoch_size : int
        Number of data points that comprise each epoch.
    
    Returns
    -------
    np.ndarray
        n_epochs x epoch_size array with the epoched data.
    """
    start_inds = epoch_inds - int(epoch_size/2)
    return rolling_window(vec, epoch_size)[start_inds, :]
     
def mean_std(arr):
    """Return mean, standard deviation."""
    return np.mean(arr), np.std(arr)
                     
def rolling_func(arr, f=np.all, window=3, right_fill=None):
    """Divide an array into rolling windows along the -1 axis, 
    and apply a function over the values in each window.
    
    Parameters
    ----------
    arr : numpy.ndarray
        The input array.
    f : function
        The function to apply (must take an axis argument).
    window : int
        The number of values in each window.
        
    Returns
    -------
    numpy.ndarray in the shape of arr if right_fill!=None,
    or with the selected axis equal to its original length
    minus (window-1) if right_fill==None.
    
    Example
    -------
    For arr=np.array([0, 1, 2, 3, 4]), f=np.sum, window=3, right_fill=None:
    
    1) The array is split into windows: [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    2) We take the sum over each tuple to get the return array: [3, 6, 9]
    3) If right_fill=0 then the return array is [3, 6, 9, 0, 0]
    """
    arr = f(rolling_window(arr, window), axis=-1)
    if right_fill is not None:
        arr = np.append(arr, np.ones(list(arr.shape[:-1]) + [window-1]) * right_fill, axis=-1)
    return arr
    
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
def unique(values, **kwargs):
    """Return unique elements and their counts as a DataFrame."""
    if 'dropna' not in kwargs:
        kwargs['dropna'] = False
    if 'sort' not in kwargs:
        kwargs['sort'] = True
    return pd.Series(values).value_counts(**kwargs)
