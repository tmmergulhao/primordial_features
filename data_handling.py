import numpy as np

#The functions below are called in the main analysis pipeline. The user should define them according
#to their own need and preference, but their output should follow the description.

def load_data_k(filename):
    """Function to load the k-array associated with the data.

    Args:
        filename (str):A string with the path to the file.
    Returns:
        k (np.array): The array containing the k-centers associated with the data.
    """
    return np.loadtxt(filename)

def load_data(filename):
    """Function to load the the data array.

    Args:
        filename (str):A string with the path to the file.

    Returns:
        data (np.array): A 1D array containing the data. It should have the same dimension as k.
    """
    return np.loadtxt(filename)

def load_cov(filename):
    """Function to load the data covariance with dimensions len(data) X len(data).

    Args:
        filename (str):A string with the path to the file.
    """
    return np.loadtxt(filename)

def load_winfunc(filename):
    """Function to load the survey window function.

    Args:
        filename (str):A string with the path to the file.
    
    Returns (np.array): A np.array with two columns containing the separation the windowfunction
    in configuration space.
    """
    return np.loadtxt(filename).T[0:2]