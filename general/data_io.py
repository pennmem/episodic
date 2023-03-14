"""
data_io.py

Author:
    Daniel Schonhaut
    
Dependencies: 
    Python 3.6, numpy, scipy, pandas.

Description: 
    Functions for data input and output.

Last Edited: 
    11/5/18
"""
import pickle

def save_pickle(obj, fpath, verbose=True):
    """Save object as a pickle file."""
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    if verbose:
        print('Saved {}'.format(fpath))
        
def open_pickle(fpath):
    """Return object."""
    with open(fpath, 'rb') as f:
        obj = pickle.load(f)
    return obj