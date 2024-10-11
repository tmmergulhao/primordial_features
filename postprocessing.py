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
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

#===================================================================================================
#Analysis settings
#===================================================================================================
#Load the Binning Specs (change it at the .json file if necessary)
with open(os.path.join(script_dir, 'BinningPosterior.json')) as json_file:
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
'smooth_scale_1D':0.1
}

#===================================================================================================
#Analysis toolkit
#===================================================================================================
def read_chain_emcee(file_name: str, burnin_frac: float, thin: int) -> Union[np.ndarray, np.ndarray]:
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

def get_total_chain(handle: str, out_f: str, n: int, burnin_frac: float, thin: int, 
compression="gzip", compression_opts=9):
    """
    Retrieve, combine, and save MCMC chains from HDF5 files.

    Args:
        handle (str): The base name of the HDF5 files containing the MCMC chains.
        out_f (str): The output HDF5 file to save the processed chains.
        n (int): The number of chains to combine. If n is 0, reads a single chain.
        burnin_frac (float): The fraction of the initial part of the chain to discard as burn-in.
        thin (int): The thinning factor to apply to the chain.
        compression (str): Compression type for HDF5 datasets.
        compression_opts (int): Compression options for HDF5 datasets.
    """

    if n == 0:
        file_name = os.path.join(f"{handle}.h5")
        print(file_name)
        final_chain, final_logprob = read_chain_emcee(file_name, burnin_frac, thin)
    else:
        final_chain, final_logprob = None, None
        for i in range(n):
            file_name = os.path.join(f"{handle}Run_{i}.h5")
            print(file_name)
            chain, logprob = read_chain_emcee(file_name, burnin_frac, thin)
            if final_chain is None:
                final_chain, final_logprob = chain, logprob
            else:
                final_chain = np.vstack((final_chain, chain))
                final_logprob = np.hstack((final_logprob, logprob))

    with h5.File(out_f, 'w') as f:
        if np.isscalar(final_chain):
            f.create_dataset('chain', data=final_chain)
        else:
            f.create_dataset('chain', data=final_chain, compression=compression, 
            compression_opts=compression_opts)

        if np.isscalar(final_logprob):
            f.create_dataset('logprob', data=final_logprob)
        else:
            f.create_dataset('logprob', data=final_logprob, compression=compression, 
            compression_opts=compression_opts)

def load_chain(file_path: str) -> np.ndarray:
    """Load the MCMC chain from an HDF5 file."""
    try:
        with h5.File(file_path, 'r') as file:
            chain = np.array(file.get("chain")).T
        return chain
    except Exception as e:
        print(f"Problem loading the file {file_path}: {e}")
        raise

def BinnedChain(handle_list: List[str], binning_limits: List[Union[float, float]], 
    file_output: str, binning_id: int, freq_bin: int = 10) -> None:
    """
    Bin the MCMC samples along the feature frequency.

    Args:
        handle_list (List[str]): A list of strings with the chain files to open.
        file_output (str): The handle to be used to name the output file.
        binning_limits (List[Tuple[float, float]]): A list of tuples with the frequency range 
        associated
         with the input files.
        omega_bin (int, optional): The frequency bin size. Defaults to 10.
        verbose (bool, optional): Whether to give details about the binning or not. Defaults to 
        False.
        freq_column (int, optional): The index of the frequency in the chains. Defaults to 3.

    Returns:
        None: This function does not return any value. It saves the results in an HDF5 file.
    """
    binned_posterior = {}

    with h5.File(file_output,'w') as out_f:
        for chain_file, [binning_min, binning_max] in zip(handle_list, binning_limits):
            #array used to bin the posterior
            binning_edges  = np.arange(binning_min, binning_max + freq_bin, freq_bin)

            #generate the keys to save the binned posterior
            keys = lambda i: f'[{binning_edges[i-1]},{binning_edges[i]}]'
            binned_posterior.update({keys(i):[]for i in range(1,len(binning_edges))})
            
            #load the chain
            chain = load_chain(chain_file)
            binning_samples = chain[binning_id]
            mapping = np.digitize(binning_samples,binning_edges)
            for sample_i,bin_it_belongs in tqdm(enumerate(mapping)):
                binned_posterior[keys(bin_it_belongs)].append(chain.T[sample_i])
        # Save the binned posterior to the HDF5 file
        for bin_label, samples in binned_posterior.items():
            out_f.create_dataset(bin_label, data=np.array(samples),
            compression='gzip', compression_opts=9)

def compute_credible_intervals(samples):
    """
    Compute the credible intervals for MCMC samples.

    Args:
        samples (np.ndarray): A 2D array of MCMC samples, where each row represents a parameter.

    Returns:
        tuple: Three arrays representing the 1σ, 2σ, and 3σ credible intervals.
    """
    sorted_samples = np.sort(samples)
    one_sigma = np.percentile(sorted_samples, [15.865, 84.135], axis=1)
    two_sigma = np.percentile(sorted_samples, [2.275, 97.725], axis=1)
    three_sigma = np.percentile(sorted_samples, [0.135, 99.865], axis=1)
    
    return one_sigma, two_sigma, three_sigma

def compute_mean(samples):
    """
    Compute the mean of MCMC samples.

    Args:
        samples (np.ndarray): A 2D array of MCMC samples, where each row represents a parameter.

    Returns:
        np.ndarray: An array of means for each parameter.
    """
    return np.mean(samples, axis=1)

def compute_std(samples):
    """
    Compute the standard deviation of MCMC samples.

    Args:
        samples (np.ndarray): A 2D array of MCMC samples, where each row represents a parameter.

    Returns:
        np.ndarray: An array of standard deviations for each parameter.
    """
    return np.std(samples, axis=1)

def compute_abs_credible(samples):
    """
    Compute the absolute credible intervals for MCMC samples.

    Args:
        samples (np.ndarray): A 2D array of MCMC samples, where each row represents a parameter.

    Returns:
        tuple: Three arrays representing the absolute 1σ, 2σ, and 3σ credible intervals.
    """
    abs_samples = np.abs(samples)
    a_star_1sigma = np.percentile(abs_samples, 68.27, axis=1)
    a_star_2sigma = np.percentile(abs_samples, 95.45, axis=1)
    a_star_3sigma = np.percentile(abs_samples, 99.73, axis=1)
    
    return a_star_1sigma, a_star_2sigma, a_star_3sigma

def compute_statistics(file_input: str, file_output: str = None):
    """
    Compute statistics from the binned MCMC samples and save them to an HDF5 file or return them.

    Args:
        file_input (str): The HDF5 file containing the binned MCMC samples.
        file_output (str, optional): The HDF5 file to save the computed statistics. If not provided, returns the results.

    Returns:
        dict: A dictionary containing computed statistics for each bin if no file_output is provided.
    """
    statistics = {}

    with h5.File(file_input, 'r') as f_in:
        for bin_label in f_in.keys():
            samples = f_in[bin_label][:].T
            
            # Compute statistics
            one_sigma, two_sigma, three_sigma = compute_credible_intervals(samples)
            mean = compute_mean(samples)
            std = compute_std(samples)
            a_star_1sigma, a_star_2sigma, a_star_3sigma = compute_abs_credible(samples)
            
            # Store results in the dictionary
            statistics[bin_label] = {
                'mean': mean,
                'std': std,
                'credible_intervals': {
                    '1_sigma': one_sigma,
                    '2_sigma': two_sigma,
                    '3_sigma': three_sigma
                },
                'abs_credible': {
                    '1_sigma': a_star_1sigma,
                    '2_sigma': a_star_2sigma,
                    '3_sigma': a_star_3sigma
                }
            }

    if file_output:
        with h5.File(file_output, 'w') as f_out:
            for bin_label, stats in statistics.items():
                grp = f_out.create_group(bin_label)
                grp.create_dataset('mean', data=stats['mean'])
                grp.create_dataset('std', data=stats['std'])
                grp.create_dataset('credible_intervals/1_sigma', data=stats['credible_intervals']['1_sigma'])
                grp.create_dataset('credible_intervals/2_sigma', data=stats['credible_intervals']['2_sigma'])
                grp.create_dataset('credible_intervals/3_sigma', data=stats['credible_intervals']['3_sigma'])
                grp.create_dataset('abs_credible/1_sigma', data=stats['abs_credible']['1_sigma'])
                grp.create_dataset('abs_credible/2_sigma', data=stats['abs_credible']['2_sigma'])
                grp.create_dataset('abs_credible/3_sigma', data=stats['abs_credible']['3_sigma'])
    else:
        return statistics

    
if __name__ == '__main__':
    n = 1
    burn_in = 0.2
    for i in range(1,10):
        f_in = "/home/tmergulhao/primordial_features/chains/lin_range1_desi_survey_catalogs_Y1_mocks_SecondGenMocks_EZmock_desipipe_v1_ffa_baseline_2pt_mock{}_pk_pkpoles_QSO_combined_z0.8-2.1_d0.001.txt_".format(i)
        f_out = "/home/tmergulhao/primordial_features/chains/lin_range1_desi_survey_catalogs_Y1_mocks_SecondGenMocks_EZmock_desipipe_v1_ffa_baseline_2pt_mock{}_pk_pkpoles_QSO_combined_z0.8-2.1_d0.001.h5".format(i)
        get_total_chain(f_in,f_out,n,burnin_frac=burn_in, thin = 5)
        f_in = f_out
        f_out = "/home/tmergulhao/primordial_features/chains/binned_lin_range1_desi_survey_catalogs_Y1_mocks_SecondGenMocks_EZmock_desipipe_v1_ffa_baseline_2pt_mock{}_pk_pkpoles_QSO_combined_z0.8-2.1_d0.001.h5".format(i)
        BinnedChain([f_in],[[100,900]],f_out,binning_id = 11, freq_bin=10)
        f_in = f_out
        f_out = "/home/tmergulhao/primordial_features/chains/stats_lin_range1_desi_survey_catalogs_Y1_mocks_SecondGenMocks_EZmock_desipipe_v1_ffa_baseline_2pt_mock{}_pk_pkpoles_QSO_combined_z0.8-2.1_d0.001.h5".format(i)
        compute_and_save_statistics(f_in,f_out)