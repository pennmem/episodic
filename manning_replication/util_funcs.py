"""
util_funcs.py

Author:
    Daniel Schonhaut
    
Dependencies: 
    Python 3.6

Description: 
    A group of simple utility functions that extend basic Python functionality.

Last Edited: 
    7/23/18
"""

def globr(search_dir, search_terms, return_type='all'):
    """Recursively search a directory for 1+ glob strings.
    
    Parameters
    ----------
    search_dir : str
        Directory that will be recursively searched. 
    search_terms : list
        List of 1+ glob strings to iterate over.
    return_type : str, optional
        Default returns both file and dir matches.
        'f' returns files only, 'd' returns directories only.
        
    Returns
    -------
    dict
        Keys are the search terms that were passed as inputs.
        Values are lists of file paths that match each search term, 
        respectively.
        
    """
    output = {}
    for search_term in search_terms:
        hits = glob.glob(os.path.join(search_dir, '**', search_term), recursive=True)
        if return_type[0] == 'f':
            output[search_term] = [f for f in hits if os.path.isfile(f)]
        elif return_type[0] == 'd':
            output[search_term] = [f for f in hits if os.path.isdir(f)]
        else:
            output[search_term] = hits
    return output