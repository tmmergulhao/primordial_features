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
        final_chain, final_logprob = read_chain_emcee(file_name, burnin_frac, thin)
    else:
        final_chain, final_logprob = None, None
        for i in range(n):
            file_name = os.path.join(final_dir, f"{handle}Run_{i}.h5")
            print(file_name)
            chain, logprob = read_chain_emcee(file_name, burnin_frac, thin)
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

def BinnedPosterior(handle_list: List[str], binning_limits: List[Union[float, float]], 
    file_output: str, binning_id: int, freq_bin: int = 10, 
    verbose: bool = False) -> None:
    """
    Bin the MCMC samples along the feature frequency and then make a histogram on the sampled
    values of amplitude to get the posterior.

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
            
#===================================================================================================
if __name__ == '__main__':

    handle_list = ['/home/tmergulhao/primordial_features/chains/total_lin_range1_pk0_QSO_mock_1_dk_0.001_kmin_0.005_kmax_0.22.h5']
    range_limits = [[100,900]]
    file_output = '/home/tmergulhao/primordial_features/chains/binned_posterior_total_lin_range1_pk0_QSO_mock_5_dk_0.001_kmin_0.005_kmax_0.22.h5'
    omega_bin = 10
    binning_axis = 11
    BinnedPosterior(handle_list,range_limits,file_output,binning_axis)