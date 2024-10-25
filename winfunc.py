# -*- coding: utf-8 -*-
# Author: Thiago Mergulhão - University of Edinburgh
# Date: 25/09/2024
# Description: This script estimates the window function of a galaxy survey data using the randoms 
# catalogue.

import numpy as np
from Corrfunc.theory import DDsmu
import sys, os, json
from scipy.special import eval_legendre
from dotenv import load_dotenv
import argparse
import logging
from mpi4py import MPI

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run MCMC analysis with different setups.')
parser.add_argument('--env', type=str, required=True, 
help='Path to the .env file for the analysis setup')
args = parser.parse_args()

# Initialize the logger only for rank 0
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    logging.basicConfig(level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
else:
    logger = None  # Do not initialize logging for other ranks

# Load environment variables from the specified .env file
load_dotenv(args.env)

# Integers
Nmu_bins = int(os.getenv("Nmu_bins"))
n_low    = int(os.getenv("n_low"))
n_mid    = int(os.getenv("n_mid"))
n_high   = int(os.getenv("n_high"))
s_min    = int(os.getenv("s_min"))
s_max    = int(os.getenv("s_max"))
Ns_bins  = int(os.getenv("Ns_bins"))
cut1     = int(os.getenv("cut1"))
cut2     = int(os.getenv("cut2"))

# Strings
out_name    = os.getenv("out_name")
randoms     = os.getenv("randoms")

# Floats
fraction_low  = float(os.getenv("fraction_low"))
fraction_mid  = float(os.getenv("fraction_mid"))
fraction_high = float(os.getenv("fraction_high"))
ells = [0, 2, 4]

# Log the loaded configuration
if rank == 0 and logger is not None:
    logger.info(f'Nmu_bins = {Nmu_bins}')
    logger.info(f'n_low = {n_low}')
    logger.info(f'n_mid = {n_mid}')
    logger.info(f'n_high = {n_high}')
    logger.info(f's_min = {s_min}')
    logger.info(f's_max = {s_max}')
    logger.info(f'cut1 = {cut1}')
    logger.info(f'cut2 = {cut2}')
    logger.info(f'Randoms catalogue loaded: {randoms}')

def RR_s(result: np.ndarray, i: int, nmu_bins: int) -> np.ndarray:
    """
    Process the output of the DDsmu function to return the number of 
    pairs per bin of mu associated with the i-th bin of s.

    Parameters:
        result (np.ndarray): Output from the DDsmu function.
        i (int): Index of the s bin.
        nmu_bins (int): Number of mu bins.
        
    Returns:
        np.ndarray: Array of pair counts for the i-th s bin.
    """
    this_s_bin = result[i * nmu_bins : (i + 1) * nmu_bins]
    output = [x[4] * x[5] for x in this_s_bin]
    return np.array(output)

def compute_Q_ell(DDsmu_output: np.ndarray, s_centers: np.ndarray, mu_centers: np.ndarray,
 ells: list = [0, 2, 4]) -> np.ndarray:
    """
    Compute the Q_ell moments for a given output of the DDsmu function.

    Parameters:
        DDsmu_output (np.ndarray): Output from the DDsmu function.
        s_centers (np.ndarray): Centers of the s bins.
        mu_centers (np.ndarray): Centers of the mu bins.
        ells (list): List of ell values to compute.
        
    Returns:
        np.ndarray: Array of computed Q_ell values.
    """
    Q_ell = np.zeros((len(ells), len(s_centers)))
    
    for ell_index, this_ell in enumerate(ells):
        # Weight of the spherical average
        w_l = 0.5 * (2 * this_ell + 1)

        for i in range(len(s_centers)):
            Q_ell[ell_index, i] = np.sum(RR_s(DDsmu_output, i, Nmu_bins) * \
                eval_legendre(this_ell, mu_centers)) * w_l

    for j in range(len(ells)):
        Q_ell[j, :] = Q_ell[j, :] / s_centers ** 3

    return Q_ell

def estimate_win(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, n: int, s_bins: np.ndarray, 
mu_centers: np.ndarray, frac: float, weights: np.ndarray, ells: list = [0, 2, 4]) -> np.ndarray:
    """
    Estimate the survey window function by averaging over n random samples.

    Parameters:
        X (np.ndarray): X coordinates of the particles.
        Y (np.ndarray): Y coordinates of the particles.
        Z (np.ndarray): Z coordinates of the particles.
        n (int): Number of iterations.
        s_bins (np.ndarray): Array defining s bins.
        mu_centers (np.ndarray): Centers of the mu bins.
        frac (float): Fraction of random samples to consider.
        weights (np.ndarray): Weights of the particles.
        ells (list): List of ell values to compute.
        
    Returns:
        np.ndarray: Averaged Q_ell values over n iterations.
    """
    npoints = X.shape[0]
    s_ctrs = 0.5 * (s_bins[1:] + s_bins[:-1])
    output = 0

    for i in range(n):
        index1 = np.random.choice(range(npoints), int(npoints * frac), replace=False)
        this_result = DDsmu(1, 20, s_bins, 1, Nmu_bins, 
                            X[index1], Y[index1], Z[index1], weights1=weights[index1],
                            weight_type='pair_product', periodic=False, verbose=False)
        output += compute_Q_ell(this_result, s_ctrs, mu_centers, ells=ells).T

    return output / n

def Compute(plot: bool = False):
    """
    Main function to perform MCMC analysis and compute Q_ell for different s ranges.

    Parameters:
        plot (bool): Flag to enable plotting of results.
    """
    s_array = np.logspace(np.log10(s_min), np.log10(s_max), Ns_bins)
    mask_low = np.where(s_array <= cut1)[0]
    mask_mid = np.where((s_array >= cut1) & (s_array < cut2))[0]
    mask_high = np.where(s_array >= cut2)[0]

    s_low = s_array[mask_low]
    s_low_ctrs = 0.5 * (s_low[1:] + s_low[:-1])

    s_mid = s_array[mask_mid[0] - 2 : mask_mid[-1]]
    s_mid_ctrs = 0.5 * (s_mid[1:] + s_mid[:-1])

    s_high = s_array[mask_high[0] - 3 :]
    s_high_ctrs = 0.5 * (s_high[1:] + s_high[:-1])
    
    mu_bins = np.linspace(0, 1, Nmu_bins + 1)
    mu_centers = 0.5 * (mu_bins[1:] + mu_bins[:-1])

    cat = np.load(randoms)
    if cat.shape[0] == 4:
        cat = cat.T

    # Extract X, Y, Z, FKP
    X = cat[:, 0]
    Y = cat[:, 1]
    Z = cat[:, 2]
    FKP = cat[:, 3]

    if rank == 0 and logger is not None:
        logger.info(f'Number of objects: {cat.shape[0]}')
        logger.info('Computing the low-s range...')
    Q_ell_low = estimate_win(X, Y, Z, n_low, s_low, mu_centers, fraction_low, FKP)
    
    if rank == 0 and logger is not None:
        logger.info('Done!')
        logger.info('Computing the mid-s range...')
    Q_ell_mid = estimate_win(X, Y, Z, n_mid, s_mid, mu_centers, fraction_mid, FKP)
    alpha1 = Q_ell_low[-1, 0] / Q_ell_mid[0, 0]
    Q_ell_mid = Q_ell_mid * alpha1

    if rank == 0 and logger is not None:
        logger.info('Done!')
        logger.info('Computing the high-s range...')

    Q_ell_high = estimate_win(X, Y, Z, n_high, s_high, mu_centers, fraction_high, FKP)
    alpha2 = Q_ell_mid[-1, 0] / Q_ell_high[0, 0]
    Q_ell_high = Q_ell_high * alpha2

    if rank == 0 and logger is not None:
        logger.info('Done!')
        logger.info('Saving the results...')

    my_result = np.vstack((Q_ell_low, Q_ell_mid[1:], Q_ell_high[1:]))
    s_final = np.hstack((s_low_ctrs, s_mid_ctrs[1:], s_high_ctrs[1:]))
    output = np.column_stack((s_final, my_result))
    np.save(out_name + '.npy', output)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for index, this_ell in enumerate([0, 2, 4]):
            plt.semilogx(s_final, output[:, index + 1], label=r"$\ell = {}$".format(this_ell))
        plt.legend()
        plt.grid()
        plt.savefig(out_name + ".pdf")

if __name__ == '__main__':
    Compute(plot=True)
