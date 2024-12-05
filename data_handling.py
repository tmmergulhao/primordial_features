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

    def load_data(self, filename):
        """
        Load the data array and apply a filter mask based on kmin and kmax.

        Args:
            filename (str): Path to the file.

        Returns:
            tuple: Filtered k array and p0 array.
        """
        _, k, _, p0, _, _ = np.genfromtxt(filename, dtype=complex, skip_header=24).T
        k, p0 = k.real, p0.real

        if self.kmin is not None or self.kmax is not None:
            mask = np.ones_like(k, dtype=bool)
            if self.kmin is not None:
                mask &= (k >= self.kmin)
            if self.kmax is not None:
                mask &= (k <= self.kmax)
            k, p0 = k[mask], p0[mask]

        return k, p0

    def load_cov(self, filename, k):
        """
        Load the data covariance matrix and apply a filter mask based on kmin and kmax.

        Args:
            filename (str): Path to the file.
            k (np.array): The k array after applying the mask.

        Returns:
            np.array: Filtered covariance matrix.
        """
        cov = np.loadtxt(filename)
        if self.kmin is not None or self.kmax is not None:
            mask = np.ones(len(k), dtype=bool)
            if self.kmin is not None:
                mask &= (k >= self.kmin)
            if self.kmax is not None:
                mask &= (k <= self.kmax)

            indices = np.where(mask)[0]
            cov = cov[np.ix_(indices, indices)]

        return cov

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