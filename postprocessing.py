#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================
# Created By  : Thiago Mergulhão - University of Edinburgh
# Created Date: 2022-02-15 10:28:29
# ==================================================================================================
"""This code define the functions needed to analyse the primordial features chains
"""
# ==================================================================================================
# Imports
# ==================================================================================================
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import emcee, os
import h5py as h5
import json, sys
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad
from getdist import MCSamples
from typing import List, Dict, Any, Optional, Callable, Union

#===================================================================================================
#Analysis settings
#===================================================================================================
#Load the Binning Specs (change it at the .json file if necessary)
with open('BinningPosterior.json') as json_file:
    BinningSpecs = json.load(json_file)

#Specs for binning the posterior
A_min = BinningSpecs["A_min"]
A_max = BinningSpecs ["A_max"]
A_bin = BinningSpecs["A_bin"]
A_array = np.arange(A_min, A_max + A_bin, A_bin)
A_ctrs = 0.5*(A_array[1:]+A_array[:-1])

#Specs for obtaining the Credible intervals
A_abs_bin = BinningSpecs["A_abs_bin"]
A_abs_min = 0
A_abs_max = np.max([abs(A_max),abs(A_min)])
A_abs_array = np.arange(A_abs_min, A_abs_max + A_abs_bin, A_abs_bin)
A_abs_ctrs = 0.5*(A_abs_array[1:]+A_abs_array[:-1])

settings_getdist = {
'ignore_rows':0,
'fine_bins':2000,
'fine_bins_2D':2000, 
'smooth_scale_1D':0.01
}

#===================================================================================================
#Analysis toolkit
#===================================================================================================
def read_chain(file_name: str, burnin_frac: float, thin: int) -> Union[np.ndarray, np.ndarray]:
    """
    Read an MCMC chain from an HDF5 file, applying burn-in and thinning.

    Args:
        file_name (str): The name of the HDF5 file containing the MCMC chain.
        burnin_frac (float): The fraction of the initial part of the chain to discard as burn-in.
        thin (int): The thinning factor to apply to the chain.

    Returns:
        np.ndarray: The MCMC chain after applying burn-in and thinning.
        np.ndarray: The log-probabilities corresponding to the chain.
    """
    backend = emcee.backends.HDFBackend(file_name, read_only=True)
    chain = backend.get_chain(flat=False)
    chain_size = chain.shape[0]
    burnin = int(burnin_frac * chain_size)
    logprob = backend.get_log_prob(flat=True, discard=burnin, thin=thin)
    chain = backend.get_chain(flat=True, discard=burnin, thin=thin)
    return chain, logprob

def get_total_chain(handle: str, n: int, burnin_frac: float, thin: int, 
dir: Optional[str] = None) -> Union[np.ndarray, np.ndarray]:
    """
    Retrieve and combine MCMC chains from HDF5 files.

    Args:
        handle (str): The base name of the HDF5 files containing the MCMC chains.
        n (int): The number of chains to combine. If n is 0, reads a single chain.
        burnin_frac (float): The fraction of the initial part of the chain to discard as burn-in.
        thin (int): The thinning factor to apply to the chain.
        dir (str, optional): An optional subdirectory within the chains directory where the HDF5 
        files are located.

    Returns:
        np.ndarray: The combined MCMC chain after applying burn-in and thinning.
        np.ndarray: The combined log-probabilities corresponding to the chain.
    """
    final_dir = 'chains/'

    if n == 0:
        file_name = os.path.join(final_dir, f"{handle}.h5")
        print(file_name)
        final_chain, final_logprob = read_chain(file_name, burnin_frac, thin)
    else:
        final_chain, final_logprob = None, None
        for i in range(n):
            file_name = os.path.join(final_dir, f"{handle}_Run_{i}.h5")
            print(file_name)
            chain, logprob = read_chain(file_name, burnin_frac, thin)
            if final_chain is None:
                final_chain, final_logprob = chain, logprob
            else:
                final_chain = np.vstack((final_chain, chain))
                final_logprob = np.hstack((final_logprob, logprob))
    return final_chain, final_logprob

def load_chain(file_path: str) -> np.ndarray:
    """Load the MCMC chain from an HDF5 file."""
    try:
        with h5.File(file_path, 'r') as file:
            chain = np.array(file.get("chain")).T
        return chain
    except Exception as e:
        print(f"Problem loading the file {file_path}: {e}")
        raise

def save_dataset(out_f, name: str, data, compression: str = "gzip", compression_opts: int = 9):
    """Save a dataset to the HDF5 file."""
    if np.isscalar(data):
        out_f.create_dataset(name, data=data)
    else:
        out_f.create_dataset(name, data=data, compression=compression, compression_opts=compression_opts)

def compute_posterior(samples: np.ndarray) -> np.ndarray:
    """Compute the posterior distribution for the amplitude."""
    samples = MCSamples(samples=samples, names=["A"], labels=["A"], settings=settings_getdist)
    posterior = samples.get1DDensity("A")
    posterior_discrete = abs(posterior(A_ctrs))
    posterior_discrete /= np.sum(posterior_discrete * A_bin)
    return posterior_discrete

def draw_samples(posterior_discrete: np.ndarray, N: int = 1000000) -> np.ndarray:
    """Draw samples from the posterior distribution."""
    return np.random.choice(A_ctrs, size=N, p=posterior_discrete * A_bin, replace=True)

def compute_statistics(samples: np.ndarray) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
     float, float, float]:
    """Compute statistics from the samples."""
    sigma_ranges = [
        np.quantile(samples, [0.15865525393149998, 0.8413447460685]),
        np.quantile(samples, [0.022750131948179, 0.977249868051821]),
        np.quantile(samples, [0.00134989803163, 0.99865010196837]),
        np.quantile(samples, [0.000031671241833, 0.999968328758167])
    ]
    median = np.median(samples)
    average = np.average(samples)
    variance = np.var(samples)
    return sigma_ranges[0], sigma_ranges[1], sigma_ranges[2],sigma_ranges[3],median,average,variance

def GetCDFThresholdPoint_1D(posterior, threshold):
    """Compute the credible interval on the primordial feature amplitude for different credible 
    regions. More precisely, this function computes the value of A associated with:
                                    P(abs(A_lin)<=A) = threshold,
    for a given threshold.

    Args:
        posterior (np.array): The y-axis of the posterior.
        threshold (float or list): A list or float associated with the credible percentage.

    Returns:
        float or list: The value of A associated with the credible region. The output is the same
        as the input.
    """
    #Interporlate the posterior
    interped_posterior = UnivariateSpline(A_ctrs, posterior, ext = 3)
    
    #List for the CDF values
    CDF_result = []
    
    #Variable to storage the cumulative sum
    cumsum = 0
    
    #Variable to storage the previous bin limit
    previous_step = 0
    
    #Number of iterations
    n_it = int(A_abs_max/A_abs_bin)
    
    #List of the A values associated with the CDF
    A_axis = []
    
    for i in range(0, n_it-1):
        new_abs_step = i*A_abs_bin
        
        #Get the integration ranges
        range_int1 = [-new_abs_step, -previous_step]
        range_int2 = [previous_step, new_abs_step]

        #Perform the integral in each range and sum them
        try:
            int1 = quad(interped_posterior,range_int1[0], range_int1[1])[0]
            int2 = quad(interped_posterior,range_int2[0], range_int2[1])[0]
        except:
            print("Problem with integration range when computing the CDF!")            
            print(range_int1)
            sys.exit(-1)
        cumsum += int1 + int2
        
        #Storage the CDF for this step
        CDF_result.append(cumsum)
        
        #Storage the A value for this step
        A_axis.append(0.5*(previous_step+new_abs_step))
        previous_step = new_abs_step

    inv_CDF_interped = interp1d(CDF_result, A_axis)
    if isinstance(threshold, list):
        output = []
        for x in threshold:
            try:
                output.append(inv_CDF_interped(x))
            except:
                print("Error saturated the maximum allowed value!")
                output.append(A_abs_max)
        return output
    else:
        return inv_CDF_interped(threshold)

def GetBinnedPosterior_1D(handle_list: List[str], range_limits: List[Union[float, float]], 
    file_output: str, dir_out: str, param_map:Dict, omega_bin: int = 10, 
    verbose: bool = False) -> None:
    """
    Bin the MCMC samples along the feature frequency and then make a histogram on the sampled
    values of amplitude to get the posterior.

    Args:
        handle_list (List[str]): A list of strings with the chain files to open.
        file_output (str): The handle to be used to name the output file.
        range_limits (List[Tuple[float, float]]): A list of tuples with the frequency range 
        associated
         with the input files.
        omega_bin (int, optional): The frequency bin size. Defaults to 10.
        verbose (bool, optional): Whether to give details about the binning or not. Defaults to 
        False.
        freq_column (int, optional): The index of the frequency in the chains. Defaults to 3.

    Returns:
        None: This function does not return any value. It saves the results in an HDF5 file.
    """
    with h5.File(dir_out+file_output,'w') as out_f:
        # Arrays to store the results
        A_1sigma, A_2sigma, A_3sigma, A_4sigma = [], [], [], []
        A_median, A_average, A_var = [], [], []
        A_cred_1sigma, A_cred_2sigma, A_cred_3sigma = [],[],[]
        n_1sigma, n_2sigma, n_3sigma, n_4sigma = 0, 0, 0, 0

        for range_index, this_handle in enumerate(handle_list):
            if verbose:
                print(f"Analysing range {range_limits[range_index]}")
            
            this_chain = load_chain(dir_out + this_handle)
            omega_min, omega_max = range_limits[range_index]
            omega_bins = np.arange(omega_min, omega_max + omega_bin, omega_bin)
            omega_ctrs = 0.5 * (omega_bins[1:] + omega_bins[:-1])

            for i in range(1, len(omega_bins)):
                bin_handle = f"[{omega_bins[i-1]}, {omega_bins[i]}]"
                if verbose:
                    print(f"Looking at bin: {bin_handle}")

                first_mask = (this_chain[param_map['omega']] >= omega_bins[i-1]) & \
                    (this_chain[param_map['omega']] < omega_bins[i])
                masked_chain = this_chain.T[first_mask]
                sampled_amplitudes = masked_chain.T[param_map['amplitude']]
                if verbose:
                    print(f"Samples inside that bin: {masked_chain.shape[0]}")

                try:
                    posterior_discrete = compute_posterior(sampled_amplitudes)
                    posterior_output = np.vstack((A_ctrs, posterior_discrete))
                    save_dataset(out_f, f"posterior:{bin_handle}", posterior_output)

                    new_samples = draw_samples(posterior_discrete)
                    sigma_ranges = compute_statistics(new_samples)
                    A_1sigma.append(sigma_ranges[0])
                    A_2sigma.append(sigma_ranges[1])
                    A_3sigma.append(sigma_ranges[2])
                    A_4sigma.append(sigma_ranges[3])
                    A_median.append(sigma_ranges[4])
                    A_average.append(sigma_ranges[5])
                    A_var.append(sigma_ranges[6])
                except:
                    A_cred_1sigma.append(0)
                    A_cred_2sigma.append(0)
                    A_cred_3sigma.append(0)
                    continue

                if not ((0 > sigma_ranges[0][0]) & (0 < sigma_ranges[0][1])):
                    n_1sigma += 1
                if not ((0 > sigma_ranges[1][0]) & (0 < sigma_ranges[1][1])):
                    n_2sigma += 1
                if not ((0 > sigma_ranges[2][0]) & (0 < sigma_ranges[2][1])):
                    n_3sigma += 1
                if not ((0 > sigma_ranges[3][0]) & (0 < sigma_ranges[3][1])):
                    n_4sigma += 1

                credible = GetCDFThresholdPoint_1D(posterior_discrete, [0.682689, 0.954499, 0.997300203])
                A_cred_1sigma.append(credible[0])
                A_cred_2sigma.append(credible[1])
                A_cred_3sigma.append(credible[2])

        save_dataset(out_f, "n_1sigma", n_1sigma)
        save_dataset(out_f, "n_2sigma", n_2sigma)
        save_dataset(out_f, "n_3sigma", n_3sigma)
        save_dataset(out_f, "n_4sigma", n_4sigma)
        save_dataset(out_f, "1sigma", np.asarray(A_1sigma))
        save_dataset(out_f, "2sigma", np.asarray(A_2sigma))
        save_dataset(out_f, "3sigma", np.asarray(A_3sigma))
        save_dataset(out_f, "4sigma", np.asarray(A_4sigma))
        save_dataset(out_f, "omega_ctrs", np.asarray(omega_ctrs))
        save_dataset(out_f, "omega_bin", omega_bin)
        save_dataset(out_f, "A_median", np.asarray(A_median))
        save_dataset(out_f, "A_average", np.asarray(A_average))
        save_dataset(out_f, "A_cred_1sigma", np.asarray(A_cred_1sigma))
        save_dataset(out_f, "A_cred_2sigma", np.asarray(A_cred_2sigma))
        save_dataset(out_f, "A_cred_3sigma", np.asarray(A_cred_3sigma))
        save_dataset(out_f, "A_var", np.asarray(A_var))
        out_f.close()
#===================================================================================================
if __name__ == '__main__':
    pass