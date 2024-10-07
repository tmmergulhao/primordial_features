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
    file_output: str, binning_id: int, freq_bin: int = 10) -> None:
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

def save_dataset(out_f, name: str, data, compression: str = "gzip", compression_opts: int = 9):
    """Save a dataset to the HDF5 file."""
    if np.isscalar(data):
        out_f.create_dataset(name, data=data)
    else:
        out_f.create_dataset(name, data=data, compression=compression, compression_opts=compression_opts)

def apply_burnin(fname, out_f, n = 0, burnin = 0.2, thin = 10):
    chain, logprob = get_total_chain(fname, n, burnin, thin)
    with h5.File(out_f, 'w') as f:
        save_dataset(f,'chain',chain)
        save_dataset(f,'logprob',logprob)

def compute_sigma_intervals(samples):
    # Sort the samples
    sorted_samples = np.sort(samples)
    
    # Compute the percentiles for 1σ, 2σ, 3σ, and 4σ
    one_sigma = np.percentile(sorted_samples, [15.865, 84.135])
    two_sigma = np.percentile(sorted_samples, [2.275, 97.725])
    three_sigma = np.percentile(sorted_samples, [0.135, 99.865])
    
    return one_sigma, two_sigma, three_sigma

def compute_a_star(samples):
    # Take the absolute value of the samples
    abs_samples = np.abs(samples)
    
    # Compute the percentiles for 1σ, 2σ, and 3σ
    a_star_1sigma = np.percentile(abs_samples, 68.27)
    a_star_2sigma = np.percentile(abs_samples, 95.45)
    a_star_3sigma = np.percentile(abs_samples, 99.73)
    
    return a_star_1sigma, a_star_2sigma, a_star_3sigma

def process_h5_file(input_file, output_file):
    with h5.File(input_file, 'r') as infile, h5.File(output_file, 'w') as outfile:
        for key in infile.keys():
            samples = np.stack(infile[key][:]).T[10]
            print(samples)
            sys.exit(-1)
            one_sigma, two_sigma, three_sigma = compute_sigma_intervals(samples)
            a_star_1sigma, a_star_2sigma, a_star_3sigma = compute_a_star(samples)
            
            # Save results in the output file
            results = {
                'one_sigma': one_sigma,
                'two_sigma': two_sigma,
                'three_sigma': three_sigma,
                'a_star_1sigma': a_star_1sigma,
                'a_star_2sigma': a_star_2sigma,
                'a_star_3sigma': a_star_3sigma
            }
            outfile.create_group(key)
            for result_key, result_value in results.items():
                outfile[key].create_dataset(result_key, data=result_value)
#===================================================================================================
if __name__ == '__main__':
    """
    fin = '/home/tmergulhao/primordial_features/chains/binned_posterior_lin_range1_desi_survey_catalogs_Y1_mocks_SecondGenMocks_EZmock_desipipe_v1_ffa_baseline_2pt_mock1_pk_pkpoles_QSO_combined_z0.8-2.1_d0.001.h5'
    fout = '/home/tmergulhao/primordial_features/chains/processed_lin_range1_desi_survey_catalogs_Y1_mocks_SecondGenMocks_EZmock_desipipe_v1_ffa_baseline_2pt_mock1_pk_pkpoles_QSO_combined_z0.8-2.1_d0.001.h5'
    process_h5_file(fin,fout)
    """
    for i in range(1,2):
        file = '/home/tmergulhao/primordial_features/chains/lin_range1_pk0_QSO_mock_{}_dk_0.001_kmin_0.005_kmax_0.22_'
        outf = '/home/tmergulhao/primordial_features/chains/lin_range1_pk0_QSO_mock_{}_dk_0.001_kmin_0.005_kmax_0.22'
        apply_burnin(file.format(i),outf.format(i),n=1,burnin=0.5)

        handle_list = [outf.format(i)]
        range_limits = [[100,900]]
        file_output = '/home/tmergulhao/primordial_features/chains/binned_posterior_lin_range1_pk0_QSO_mock_{}_dk_0.001_kmin_0.005_kmax_0.22'
        omega_bin = 10
        binning_axis = 11
        BinnedPosterior(handle_list,range_limits,file_output.format(i),binning_axis)