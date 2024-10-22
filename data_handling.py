import numpy as np

def compute_mask(k, KMIN=None, KMAX=None):
    """Helper function to compute the mask based on KMIN and KMAX.

    Args:
        k (np.array): The k array to apply the filter on.
        KMIN (float, optional): The minimum value of k to filter. Defaults to None.
        KMAX (float, optional): The maximum value of k to filter. Defaults to None.

    Returns:
        mask (np.array): A boolean array representing the mask for filtering.
    """
    if KMIN is not None and KMAX is not None:
        mask = (k >= float(KMIN)) & (k <= float(KMAX))
    elif KMIN is not None:
        mask = k >= float(KMIN)
    elif KMAX is not None:
        mask = k <= float(KMAX)
    else:
        mask = np.ones_like(k, dtype=bool)  # No filtering, mask includes all elements
    return mask

def load_data_k(filename, mask=None):
    """Function to load the k-array associated with the data and apply a filter mask.

    Args:
        filename (str): A string with the path to the file.
        mask (np.array, optional): A boolean array representing the filter. Defaults to None.

    Returns:
        k (np.array): The array containing the k-centers associated with the data.
    """
    k = np.loadtxt(filename)
    if mask is not None:
        return k[mask]
    return k

def load_data(filename, mask=None):
    """Function to load the data array and apply a filter mask.

    Args:
        filename (str): A string with the path to the file.
        mask (np.array, optional): A boolean array representing the filter. Defaults to None.

    Returns:
        data (np.array): A 1D array containing the filtered data. It should have 
        twice the length of k-array, as NGC and SGC are stacked.
    """
    data = np.loadtxt(filename)
    N = len(data) // 2  # Assume NGC and SGC have equal lengths, so data is twice as long

    if mask is not None:
        mask_ngc = mask[:N]
        mask_sgc = mask[:N]  # Use the same mask for SGC assuming symmetry
        return np.hstack((data[:N][mask_ngc], data[N:][mask_sgc]))

    return data

def load_cov(filename, mask=None):
    """Function to load the data covariance with dimensions 2*len(data) X 2*len(data) (the factor
    of 2 comes after combining NGC with SGC).

    Args:
        filename (str): A string with the path to the file.
        mask (np.array, optional): A boolean array representing the filter. Defaults to None.

    Returns:
        cov_filtered (np.array): The filtered covariance matrix.
    """
    cov = np.loadtxt(filename)
    N = cov.shape[0] // 2  # The length of the NGC (and SGC) section

    if mask is not None:
        cov_ngc = cov[:N, :N][mask][:, mask]  # Filter NGC block
        cov_sgc = cov[N:, N:][mask][:, mask]  # Filter SGC block

        # Create the filtered covariance matrix with zeroed-off diagonal blocks
        cov_filtered = np.zeros((cov_ngc.shape[0] + cov_sgc.shape[0], cov_ngc.shape[1] + cov_sgc.shape[1]))
        cov_filtered[:cov_ngc.shape[0], :cov_ngc.shape[1]] = cov_ngc  # NGC block
        cov_filtered[cov_ngc.shape[0]:, cov_ngc.shape[1]:] = cov_sgc  # SGC block

        return cov_filtered

    return cov


def load_winfunc(filename):
    """Function to load the survey window function.

    Args:
        filename (str):A string with the path to the file.
    
    Returns (np.array): A np.array with two columns containing the separation the windowfunction
    in configuration space.
    """
    return np.loadtxt(filename).T[0:2]