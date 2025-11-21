"""
A helper module containing utiltiy functions 
"""


def rms(data):
    """
    Compute the root mean square of an array.
    Parameters:
        data: array-like, The input data
    Returns:
        float: The root mean square of the data
    """
    import numpy as np
    data = np.asarray(data)
    return np.sqrt(np.mean(data**2))
