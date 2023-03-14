"""
helper_funcs.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
----------- 
Helper functions for common tasks with simple python classes.

Last Edited
----------- 
3/19/22
"""
from time import time
from itertools import chain, zip_longest
from collections import OrderedDict as od
import numpy as np
from scipy.stats import sem


class Timer(object):
    """I say how long things take to run."""

    def __init__(self):
        """Start the global timer."""
        self.reset()

    def __str__(self):
        """Print how long the global timer has been running."""
        msg = 'Ran in {:.1f}s'.format(self.check())
        return msg

    def check(self,
              reset=False):
        """Report the global runtime."""
        runtime = time() - self.start
        if reset:
            self.reset()
        return runtime

    def loop(self,
             key=None,
             verbose=True):
        """Report the loop runtime and reset the loop timer."""
        if not hasattr(self, 'loops'):
            self.loops = od([])
        if not hasattr(self, 'last_loop_start'):
            self.last_loop_start = self.start
        if key is None:
            key = 'loop {}'.format(len(self.loops) + 1)
        
        loop_runtime = time() - self.last_loop_start
        self.loops[key] = loop_runtime
        self.last_loop_start = time()
        if verbose:
            print('{}: {:.1f}s'.format(key, self.loops[key]))

    def reset(self):
        """Reset the global timer."""
        self.start = time()


def weave(l1, l2):
    """Interleave two lists of same or different lengths."""
    return [x for x in chain(*zip_longest(l1, l2)) if x is not None]


def invert_dict(d):
    """Invert a dictionary of string keys and list values."""
    if type(d) == dict:
        newd = {}
    else:
        newd = od([])
    for k, v in d.items():
        for x in v:
            newd[x] = k
    return newd


def str_replace(obj_in, 
                replace_vals=None):
    """Multi-string replacement for strings and lists of strings.
    
    Parameters
    ----------
    obj_in : str or list[str]
        A single string or a list of strings
        with values that you want to replace.
    replace_vals : dict or OrderedDict
        {old_value: new value, ...} in the order given.
    
    Returns 
    -------
    obj_out : str or list[str]
        Same as obj_in but with values replaced.
    """
    if isinstance(obj_in, str):
        obj_out = [obj_in]
    else:
        obj_out = obj_in.copy()

    for _i in range(len(obj_out)):
        for old_str, new_str in replace_vals.items():
            obj_out[_i] = obj_out[_i].replace(old_str, new_str)
    
    if isinstance(obj_in, str):
        obj_out = obj_out[0]
            
    return obj_out


def strip_space(in_str):
    """Strip 2+ adjoining spaces down to 1."""
    out_str = in_str.strip()
    for iSpace in range(len(in_str), 1, -1):
        search_str = ' ' * iSpace
        out_str = out_str.replace(search_str, ' ')

    return out_str


def count_pct(vals,
              decimals=1):
    """Return count_nonzero/n (percent)."""
    vals = np.array(vals)
    string = '{}/{} ({:.{_}%})'.format(np.count_nonzero(vals>0),
                                       len(vals),
                                       np.count_nonzero(vals>0)/len(vals),
                                       _=decimals)
    return string


def mean_sem(vals,
             decimals=2):
    """Return mean ± standard error."""
    string = '{:.{_}f} ± {:.{_}f}'.format(np.nanmean(vals),
                                          sem(vals, nan_policy='omit'),
                                          _=decimals)
    return string


def mean_sd(vals,
            decimals=2):
    """Return mean ± standard error."""
    string = '{:.{_}f} ± {:.{_}f}'.format(np.nanmean(vals),
                                          np.nanstd(vals),
                                          _=decimals)
    return string


def gmean_sem(vals,
              decimals=2):
    """Return geometric mean ± standard error."""
    log_vals = np.log10(vals)
    _mean = 10**np.nanmean(log_vals)
    _sem = 10**sem(log_vals, nan_policy='omit')
    string = '{:.{_}f} ± {:.{_}f}'.format(_mean, _sem, _=decimals)
    return string


def median_q(vals,
             decimals=2):
    """Return median (lower quartile, upper quartile)."""
    string = '{:.{_}f} ({:.{_}f}, {:.{_}f})'.format(np.nanmedian(vals),
                                                    *np.nanpercentile(vals, [25, 75]),
                                                    _=decimals)
    return string


def circit(val,
           prop='r',
           scale=1):
    """Solve for the properties of, and/or transform, a circle.
    
    Parameters
    ----------
    val : number > 0
        Value of the input circle property.
    prop : str
        'r' = radius
        'd' = diameter
        'a' = area
        'c' = circumference
    scale : number > 0
        Applies val *= scale to the output circle.
    
    Returns
    -------
    circle : dict
        Contains r, d, a, and c versus the input circle.
    """
    # Transform the output circle.
    val *= scale

    # Solve the circle's properties.
    if prop == 'r':
        r = val
        d = r * 2
        a = np.pi * np.square(r)
        c = 2 * np.pi * r
    elif prop == 'd':
        d = val
        r = d / 2
        a = np.pi * np.square(r)
        c = 2 * np.pi * r
    elif prop == 'a':
        a = val
        r = np.sqrt(a / np.pi)
        d = r * 2
        c = 2 * np.pi * r
    elif prop == 'c':
        c = val
        r = c / (2 * np.pi)
        d = r * 2
        a = np.pi * np.square(r)
    
    # Store the outputs.
    circle = {'r': r,
              'd': d,
              'a': a,
              'c': c}
    
    return circle
