import numpy as np
import os
import json

class DataProcessor:
    def __init__(self, kmin=None, kmax=None):
        """
        Initialize the DataProcessor with optional kmin and kmax values.

        Args:
            kmin (float, optional): Minimum k value for filtering. Defaults to None.
            kmax (float, optional): Maximum k value for filtering. Defaults to None.
        """
        self.kmin = kmin
        self.kmax = kmax
        self.mask = None  # Placeholder for the mask

    def load_data(self, filename):
        """
        Load the data array, apply the filter, and store k, p0, and mask as attributes.

        Args:
            filename (str): Path to the file.
        """
        _, k, _, p0, _, _ = np.genfromtxt(filename, dtype=complex, skip_header=24).T
        k, p0 = k.real, p0.real

        # Create a mask based on kmin and kmax
        self.mask = np.ones_like(k, dtype=bool)
        if self.kmin is not None:
            self.mask &= (k >= self.kmin)
        if self.kmax is not None:
            self.mask &= (k <= self.kmax)

        # Apply the mask
        self.k = k[self.mask]
        self.p0 = p0[self.mask]
        return self.k, self.p0
    
    def load_cov(self, filename):
        """
        Load the data covariance matrix and apply the filter mask using its outer product.

        Args:
            filename (str): Path to the file.

        Returns:
            np.array: Filtered covariance matrix.
        """
        if self.mask is None:
            raise ValueError("Mask is not defined. Please call load_data() first.")

        cov = np.loadtxt(filename)

        # Use the outer product of the mask to filter the covariance matrix
        mask_outer = np.outer(self.mask, self.mask)
        cov_filtered = cov[mask_outer].reshape(self.mask.sum(), self.mask.sum())

        return cov_filtered

def load_winfunc(filename):
    """
    Load the survey window function.

    Args:
        filename (str): Path to the file (.txt or .npy).

    Returns:
        np.array: Array with two columns containing the window function separation.
    """
    _, file_extension = os.path.splitext(filename)
    if file_extension == ".txt":
        return np.loadtxt(filename).T[0:2]
    elif file_extension == ".npy":
        return np.load(filename).T[0:2]
    else:
        raise ValueError("Unsupported file format. Please provide a .txt or .npy file.")

def load_json_to_dict(file_path):
    """
    Load a JSON file into a dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Dictionary representation of the JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
    return {}