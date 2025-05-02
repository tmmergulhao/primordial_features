import numpy as np
import os
import json
import h5py
import scipy.interpolate as interp
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

    def load_data_DESI(self, filename):
        """
        Load the data array, apply the filter, and store k, p0, and mask as attributes.

        Args:
            filename (str): Path to the file.
        """
        if 'synthetic' in filename:
            self.k,self.p0 = np.loadtxt(filename)
            return self.k, self.p0

        data = np.genfromtxt(filename, dtype=complex, skip_header=24).T
        k, p0 = data[1].real, data[3].real

        # Create a mask based on kmin and kmax
        self.mask = np.ones_like(k, dtype=bool)
        if self.kmin is not None:
            self.mask &= (k >= self.kmin)
        if self.kmax is not None:
            self.mask &= (k <= self.kmax)

        # Apply the mask
        self.original_k = k.copy()
        self.k = k[self.mask]
        self.p0 = p0[self.mask]
        return self.k, self.p0

    def load_data_BOSS(self, filename):
        """
        Load the data array, apply the filter, and store k, p0, and mask as attributes.

        Args:
            filename (str): Path to the file.
        """
        if 'synthetic' in filename:
            self.k,self.p0 = np.loadtxt(filename)
            return self.k, self.p0

        data = np.genfromtxt(filename, dtype=complex, skip_header=33).T
        k, p0 = data[1].real, data[2].real

        # Create a mask based on kmin and kmax
        self.mask = np.ones_like(k, dtype=bool)
        if self.kmin is not None:
            self.mask &= (k >= self.kmin)
        if self.kmax is not None:
            self.mask &= (k <= self.kmax)

        # Apply the mask
        self.original_k = k.copy()
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
        cov = cov[0:len(self.original_k), 0:len(self.original_k)] # only get the monopole covariance
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

def read_bingo_results(filename):
    """
    Reads the bingo results from an HDF5 file and returns:
      - k_values: 1D array of k values
      - deltaN_values: 1D array of Delta N values (from header)
      - deltaP_values: 2D array of deltaP values
    """
    with h5py.File(filename, 'r') as f:
        k_values = f['k'][:]         # 1D array
        deltaN_values = f['deltaN'][:]  # 1D array from header
        deltaP_values = f['deltaP'][:]  # 2D array of deltaP values
    return k_values, deltaN_values, deltaP_values

# Function to create interpolation on log10(k) and log10(Delta N)
def create_interpolation_function_bump(k_values, deltaN_values, deltaP_values):
    # Log10(k) and Log10(Delta N) for interpolation
    logk_values = np.log10(k_values)
    logdeltaN_values = np.log10(deltaN_values)
    
    # Dimension check
#    print(f"logk_values shape: {logk_values.shape}")
#    print(f"logdeltaN_values shape: {logdeltaN_values.shape}")
#    print(f"deltaP_values shape: {deltaP_values.shape}")
    
    # Ensure deltaP dimensions are consistent
    if deltaP_values.shape[0] != len(k_values) or deltaP_values.shape[1] != len(deltaN_values):
        raise ValueError(f"Inconsistent dimensions: deltaP shape {deltaP_values.shape}, "
                         f"k length {len(k_values)}, deltaN length {len(deltaN_values)}")
    
    # Create 2D interpolation on log10(k) and log10(Delta N)
    interp_func = interp.RectBivariateSpline(logk_values, logdeltaN_values, deltaP_values)
    
    # Return the interpolated function
    return interp_func

    
# Function for the correction to the power spectrum with a feature
def deltaPfeature_bump(k, dP, N0, deltaN, interp_func, pivot=0.05):
    
    # Amplitude modulation term
    Amp = dP / 0.025
    
    arg = k / np.exp(N0-15)
    # Interpolate deltaP using the interpolation function
    deltaP_interp = interp_func(np.log10(arg), np.log10(deltaN), grid=False)
    
    # Calculate the power spectrum with the feature
    return Amp * deltaP_interp

# Function to create interpolation on log10(k) and log10(m_sigma/H)
def create_interpolation_function_cpsc(k_values, log10_m_over_H_values, deltaP_values):
    # Log10(k) and Log10(m_sigma/H) for interpolation
    logk_values = np.log10(k_values)

    # Ensure deltaP dimensions are consistent
    if deltaP_values.shape[0] != len(k_values) or deltaP_values.shape[1] != len(log10_m_over_H_values):
        raise ValueError(f"Inconsistent dimensions: deltaP shape {deltaP_values.shape}, "
                         f"k length {len(k_values)}, deltaN length {len(log10_m_over_H_values)}")
    
    # Create 2D interpolation on log10(k) and log10(m_sigma/H)
    interp_func = interp.RectBivariateSpline(logk_values, log10_m_over_H_values, deltaP_values)
    
    # Return the interpolated function
    return interp_func

    
    
# Function for the correction to the power spectrum with a feature
def deltaPfeature_cpsc(k, dP, N0, log10_m_over_h, interp_func, pivot=0.05):
    
    # Amplitude modulation term
    Amp = dP / 0.01
    
    arg = k / np.exp(N0-0.135036080469101E+002)
    # Interpolate deltaP using the interpolation function
    deltaP_interp = interp_func(np.log10(arg), log10_m_over_h, grid=False)
    
    # Calculate the power spectrum with the feature
    return Amp * deltaP_interp