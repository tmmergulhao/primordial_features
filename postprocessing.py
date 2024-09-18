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
from re import X
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import emcee, os
import h5py as h5
import json, sys
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad
import getdist
# ==================================================================================================
# Directories
# ==================================================================================================
this_directory = os.getcwd()+"/" #save this diretory
os.chdir(os.path.dirname(os.getcwd()))#go to main directory
master_directory = os.getcwd()+"/" #save the master directory
figures_directory = this_directory+"figures/"

with open(master_directory+'dir.json') as json_file:
    dir = json.load(json_file)
results_directory = dir["results_dir"]
del dir

#===================================================================================================
#Analysis settings
#===================================================================================================
#Load the Binning Specs (change it at the .json file if necessary)
with open(this_directory+'BinningPosterior.json') as json_file:
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
print(A_abs_ctrs)
settings_getdist = {
'ignore_rows':0,
'fine_bins':2000,
'fine_bins_2D':2000, 
'smooth_scale_1D':0.1
}
#===================================================================================================
#Analysis toolkit
#===================================================================================================
def GetTotalChain(handle, n, burnin_frac, thin, dir):
    if dir is not None:
        final_dir = master_directory+"chains/"+dir+"/"

    else:
        final_dir = master_directory+"chains/"

    if n==0:
        name = final_dir+handle+'.h5'
        print(name)
        backend = emcee.backends.HDFBackend(name, read_only = True)
        chain = backend.get_chain(flat=False)
        chain_size = chain.shape[0] #Get the size of the array
        burnin = int(burnin_frac*chain_size)
        final_logprob = backend.get_log_prob(flat = True, discard = burnin, thin = thin)
        final_chain = backend.get_chain(flat=True, discard = burnin, thin = thin)
   
    else:
        for i in range(0,n):
            name = final_dir+handle+'Run_{}.h5'.format(i)
            print(name)
            backend = emcee.backends.HDFBackend(name, read_only = True)
            chain = backend.get_chain(flat=False)
            chain_size = chain.shape[0] #Get the size of the array
            burnin = int(burnin_frac*chain_size)
            chain = backend.get_chain(flat=True, discard = burnin, thin = thin)
            logprob = backend.get_log_prob(flat = True, discard = burnin, thin = thin)
            if i==0:
                final_chain = chain
                final_logprob = logprob
            else:
                final_chain = np.vstack((final_chain, chain))
                final_logprob = np.hstack((final_logprob, logprob))
    print("chi2:", -2*np.max(final_logprob))
    return final_chain, final_logprob

def chain_diagnostic(handle, burnin_frac, n, omega_min, omega_max, omega_bin, params, thin = 10, 
dir = None, omega_index = 3):

    #Create the output file
    try:
        print(results_directory+handle+"_derived_results.h5")
        f = h5.File(results_directory+handle+"_derived_results.h5", "w")
        f_chain = h5.File(results_directory+handle+"_total_chain.h5", "w")

    except:
        print("Deleting pre-existing file and creating a new one.")
        os.remove(results_directory+handle+"_derived_results.h5")
        f = h5.File(results_directory+handle+"_derived_results.h5", "w")
        f_chain = h5.File(results_directory+handle+"_total_chain.h5", "w")

    #Join all chains into a single one
    final_chain, final_logprob = GetTotalChain(handle, n, burnin_frac, thin = thin, dir = dir)

    f_chain.create_dataset('chain', data = final_chain, compression="gzip", compression_opts=9)
    f_chain.create_dataset('logprob', data = final_logprob, compression="gzip", compression_opts=9)
    f_chain.close()
    
    #save the MCMC chain and chi2

    #Define frequency bins
    omega_bins = np.arange(omega_min,omega_max + omega_bin, omega_bin)
    omega_ctrs = 0.5*(omega_bins[1:] + omega_bins[:-1])

    #Chain result for omega
    omega_chain = final_chain[:,omega_index]
    
    for this_param in list(params.keys()):
        #Arrays to storage the mean and variance
        this_median = np.zeros_like(omega_ctrs)
        this_mean   = np.zeros_like(omega_ctrs)
        this_std    = np.zeros_like(omega_ctrs)
        this_1sigma = np.zeros_like(omega_ctrs)    
        this_2sigma = np.zeros_like(omega_ctrs)
        this_3sigma = np.zeros_like(omega_ctrs)

        sampled_values = final_chain[:, params[this_param]]

        for j in range(1, len(omega_bins)):

            #Apply the mask
            omega_min = omega_bins[j-1]
            omega_max = omega_bins[j]
            mask = (omega_chain>=omega_min)&(omega_chain<omega_max)
            sampled_values_masked = sampled_values[mask]

            #Computing statistics
            this_mean[j-1]  = np.mean(sampled_values_masked)
            this_std[j-1] = np.std(sampled_values_masked)
            this_median[j-1] = np.median(sampled_values_masked)

            sigma1 = (np.percentile(sampled_values_masked, 50+34.1, axis=0)- np.percentile(sampled_values_masked, 50-34.1, axis=0))/2.
            sigma2 = (np.percentile(sampled_values_masked, 50+34.1+13.6, axis=0)- np.percentile(sampled_values_masked, 50-34.1-13.6, axis=0))/2.
            sigma3 = (np.percentile(sampled_values_masked, 50+34.1+13.6+2.1, axis=0)- np.percentile(sampled_values_masked, 50-34.1-13.6-2.1, axis=0))/2.
            
            this_1sigma[j-1] = sigma1
            this_2sigma[j-1] = sigma2
            this_3sigma[j-1] = sigma3

        label_mean = this_param+"_mean"
        label_std = this_param+"_std"
        label_1sigma = this_param+"_1sigma"
        label_2sigma = this_param+"_2sigma"
        label_3sigma = this_param+"_3sigma"

        f.create_dataset(label_mean, data = this_mean)
        f.create_dataset(label_std, data = this_std)
        f.create_dataset(label_1sigma, data = this_1sigma)
        f.create_dataset(label_2sigma, data = this_2sigma)
        f.create_dataset(label_3sigma, data = this_3sigma)

    f.create_dataset("omega_ctrs", data = omega_ctrs)
    f.create_dataset("omega_binsize", data = omega_bin)
    f.close()

def PlotParamsvsFreq(handle_list, MOCKS_iterable, params, filename_output, range_list = None):

    #iterate over the params
    for this_param in params:

        #initialize one figure per parameter
        fig, ax = plt.subplot_mosaic([["A", "B"]], tight_layout = True, figsize = (10, 8))

        #iterate over different mocks
        for this_mock in MOCKS_iterable:

            #list to storage the different files associated to the same mock
            file_list = []

            #iterate of the files
            for this_handle in handle_list:
                file_list.append(h5.File(results_directory+this_handle.format(this_mock), 'r'))
            
            for range_index, this_file in enumerate(file_list):
                this_mean = np.array(this_file[this_param+"_mean"])
                this_sigma = np.array(this_file[this_param+"_sigma"])
                this_omega_array = np.array(this_file["omega_ctrs"])

                if range_list is not None:
                    mask = (this_omega_array>range_list[range_index][0])&\
                        (this_omega_array<=range_list[range_index][1])
                    this_omega_array = this_omega_array[mask]
                    this_sigma = this_sigma[mask]
                    this_mean = this_mean[mask]

                ax["A"].plot(this_omega_array, this_mean, label = "{}".format(this_mock))
                ax["B"].plot(this_omega_array, this_sigma, label = "{}".format(this_mock))
        
        ax["A"].legend(ncol = 5)
        ax["B"].legend(ncol = 5)
        plt.savefig(figures_directory+filename_output+this_param+"_vsFreq.png")
        plt.close('all')

def GetUnifiedResults(handle_list, handle_output, MOCKS_iterable, params, range_limits = None):

    out_f = h5.File(results_directory+handle_output.format(i)+".h5", 'w')

    #iterate over the params
    for this_param in params:

        first_it_mock = 1
        #iterate over different mocks
        for this_mock in MOCKS_iterable:
        
            first_it_range = 1
            #Iterate over the freq. ranges and stack the results
            for range_index, this_handle in enumerate(handle_list):
                print(this_handle.format(this_mock))
                this_file = h5.File(results_directory+this_handle.format(this_mock), 'r')

                #If it is the first mock, initialise the output
                if first_it_range:

                    final_mean = np.array(this_file[this_param+"_mean"])
                    final_sigma = np.array(this_file[this_param+"_1sigma"])
                    final_omega = np.array(this_file["omega_ctrs"])
                    
                    if range_limits is not None:
                        mask = (final_omega>range_limits[range_index][0])&\
                            (final_omega<=range_limits[range_index][1])
                        final_omega = final_omega[mask]
                        final_sigma = final_sigma[mask]
                        final_mean = final_mean[mask]
                #If not the first iteration, stock this result with the other ones
                else:
                    this_mean = np.array(this_file[this_param+"_mean"])
                    this_sigma = np.array(this_file[this_param+"_1sigma"])
                    this_omega = np.array(this_file["omega_ctrs"])

                    if range_limits is not None:
                        mask = (this_omega>range_limits[range_index][0])&\
                            (this_omega<=range_limits[range_index][1])
                        this_omega = this_omega[mask]
                        this_sigma = this_sigma[mask]
                        this_mean = this_mean[mask]
                    final_mean = np.hstack((final_mean, this_mean))
                    final_sigma = np.hstack((final_sigma, this_sigma))
                    final_omega = np.hstack((final_omega, this_omega))
        
                this_file.close()
                first_it_range = 0

            if first_it_mock:
                stacked_mean = final_mean
                stacked_sigma = final_sigma
            else:
                stacked_mean = np.column_stack((stacked_mean, final_mean))
                stacked_sigma = np.column_stack((stacked_sigma, final_sigma))

            first_it_mock = 0


        print(stacked_mean.shape)
        print(stacked_sigma.shape)
        
        out_f.create_dataset(this_param+"_mean", data = stacked_mean)
        out_f.create_dataset(this_param+"_sigma", data = stacked_sigma)
    
    out_f.create_dataset("omega_ctrs", data = final_omega)
    out_f.close()

def GetCDFThresholdPoint(posterior):
    """Returns where the CDF reachs the confidence interval threshold.

    Args:
        posterior (np.array): The 2D normalised posterior in A_sin and A_cos.
    """

    #List to storage the CDF for each value of the radius array
    CDF_results = []

    #The integral is performed by integrating inside circles of increasing radius
    for x in A_abs:
        #Creates a null mask
        mask = np.full((len(A_ctrs), len(A_ctrs)), False)

        #Nested loop over A_sin and A_cos. Fill the mask to get values inside the circle
        for i, this_Asin in enumerate(A_ctrs):
            for j, this_Acos in enumerate(A_ctrs):
                this_Abs = np.sqrt(this_Asin**2 + this_Acos**2)            

                #Return True for point inside the circle
                if this_Abs <= x:
                    mask[i,j] = True

        CDF_results.append(np.sum(posterior[mask]*A_bin**2))
    index = np.where(np.array(CDF_results)>=threshold)[0][0] 
    return A_abs[index]

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

def GetCDFThresholdPoint_1D_getdist(posterior, norm, threshold):
    """Compute the credible interval on the primordial feature amplitude for different credible 
    regions. More precisely, this function computes the value of A associated with:
                                    P(abs(A_lin)<=A) = threshold,
    for a given threshold. Instead of making an histogram from the samples, we use the Getdist
    density module to have a smooth version of it. It usually reduces chain noise.

    Args:
        posterior (getdist 1D density or interped func): The 1D Density for the amplitude parameter.
        norm (int): The normalization for the posterior.
        threshold (float or list): A list or float associated with the credible percentage.

    Returns:
        float or list: The value of A associated with the credible region. The output is the same
        as the input.
    """
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
            int1 = quad(posterior,range_int1[0], range_int1[1])[0]/norm
            int2 = quad(posterior,range_int2[0], range_int2[1])[0]/norm
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

def GetBinnedPosterior(handle_list, file_output, range_limits,
omega_bin = 10, verbose = False, freq_column = 3):
    
    #File to storage the output
    try:
        out_f = h5.File(results_directory+file_output+".h5", 'w')
    except:
        """
        print("Deleting pre-existing file and creating a new one.")
        os.remove(results_directory+file_output+".h5")
        out_f = h5.File(results_directory+file_output+".h5", 'w')
        """
        print("Problem opening the file")
        sys.exit(-1)

    #Array to storage the confidence intervals
    A_conf = []
    
    #Iterate over the freq. ranges and stack the results
    for range_index, this_handle in enumerate(handle_list):
        print("Analysing range", range_limits[range_index])
        #Load the file associated to this freq. range and its chain
        this_file = h5.File(results_directory+this_handle, 'r')
        this_chain = np.array(this_file.get("chain")).T
        this_file.close()

        #Get the details of this freq. range
        print(range_limits[range_index])
        omega_min, omega_max = range_limits[range_index]

        #Define frequency bins - Used to split the chains
        omega_bins = np.arange(omega_min,omega_max + omega_bin, omega_bin)

        for i in tqdm(range(1, len(omega_bins))):
            #String to label the outputs
            bin_handle = "[{}, {}]".format(omega_bins[i-1],omega_bins[i])
            if verbose: print("Looking at bin:", bin_handle)

            #Get the chain elements inside this specific frequency bin
            first_mask = (this_chain[freq_column]>=omega_bins[i-1])&\
                (this_chain[freq_column]<omega_bins[i])
            masked_chain = this_chain.T[first_mask]
            
            if verbose: print("Samples inside that bin: {}".format(masked_chain.shape[0]))

            #Compute the normalised posterior for the two amplitude parameters
            posterior = np.histogram2d(masked_chain.T[1], masked_chain.T[2], bins = A_array, \
                density = True)[0]

            norm = np.sum(posterior*A_bin**2)
            if verbose: print("Normalization factor: ", norm)
            
            #save this posterior in the output file
            out_f.create_dataset("posterior:"+bin_handle, data = posterior, \
                compression="gzip", compression_opts=9)                

            #Computing p(|A| vs |A|)
            A_conf.append(GetCDFThresholdPoint(posterior))

    #End of the iterarion over all freq. ranges for a given mock. Saving the outputs
    out_f.create_dataset("ConfidenceInterval", data = np.asarray(A_conf))
    out_f.create_dataset("omega_bin", data = omega_bin)
    out_f.close()

def GetBinnedPosterior_1D(handle_list, file_output, range_limits,
omega_bin = 10, verbose = False, freq_column = 3):
    """Bin the MCMC samples along the feature frequency and then make a histogram on the sampled
    values of amplitude to get the posterior.

    Args:
        handle_list (list): A list of strings with the chain files to open
        file_output (list): A list of handles to be used to name the output files
        range_limits (list): A list of bidimensional lists with the frequency range associated with
        the input files
        omega_bin (int, optional): The frequency bin size. Defaults to 10.
        verbose (bool, optional): Wether to give details about the binning or not. Defaults to False.
        freq_column (int, optional): The index of the frequency in the chains. Defaults to 3.
    """

    #File to storage the output
    try:
        out_f = h5.File(results_directory+file_output+".h5", 'w')
    except:
        print("Problem opening the file")
        sys.exit(-1)

    #Array to storage the credible regions around the PDF peak 
    A_1sigma = []
    A_2sigma = []
    A_3sigma = []
    A_4sigma = []
    A_median = []
    A_average = []
    A_var = []

    #Array to storage the credible intervals for the amplitude
    A_cred_1sigma = []
    A_cred_2sigma = []
    A_cred_3sigma = []

    #count the number of events
    n_1sigma = 0
    n_2sigma = 0
    n_3sigma = 0
    n_4sigma = 0

    #Iterate over the freq. ranges and stack the results
    for range_index, this_handle in enumerate(handle_list):
        print("Analysing range", range_limits[range_index])
        #Load the file associated to this freq. range and its chain
        this_file = h5.File(results_directory+this_handle, 'r')
        this_chain = np.array(this_file.get("chain")).T
        this_file.close()

        #Get the details of this freq. range
        omega_min, omega_max = range_limits[range_index]

        #Define frequency bins - Used to split the chains
        omega_bins = np.arange(omega_min,omega_max + omega_bin, omega_bin)
        omega_ctrs = 0.5*(omega_bins[1:] + omega_bins[:-1])

        for i in tqdm(range(1, len(omega_bins))):
            #String to label the outputs
            bin_handle = "[{}, {}]".format(omega_bins[i-1],omega_bins[i])
            if verbose: print("Looking at bin:", bin_handle)

            #Get the chain elements inside this specific frequency bin
            first_mask = (this_chain[freq_column]>=omega_bins[i-1])&\
                (this_chain[freq_column]<omega_bins[i])
            masked_chain = this_chain.T[first_mask]
            
            if verbose: print("Samples inside that bin: {}".format(masked_chain.shape[0]))

            #Initialise getdist to get the marg. posterior for the feature amplitude
            samples = getdist.MCSamples(samples = masked_chain.T[1], names = ["A"],
             labels = ["A"], settings = settings_getdist)
            bare_posterior = samples.get1DDensity("A")
            norm = quad(bare_posterior, -1, 1, limit = 150)[0]

            #Compute the posterior at some points to give as output
            posterior_discrete = abs(bare_posterior(A_ctrs))
            posterior_discrete/= np.sum(posterior_discrete*A_bin)
            print(np.min(posterior_discrete))
            posterior_output = np.vstack((A_ctrs,  posterior_discrete)) 

            out_f.create_dataset("posterior:"+bin_handle, data = posterior_output, \
                compression="gzip", compression_opts=9)                

            #Get new samples from the PDF to compute the statistics
            N = 1000000 #number of samples to draw
            new_samples = np.random.choice(A_ctrs, size=N, p=posterior_discrete*A_bin, replace=True)
            this_1sigma_range = np.quantile(new_samples, [0.15865525393149998, 0.8413447460685])
            this_2sigma_range = np.quantile(new_samples, [0.022750131948179, 0.977249868051821])
            this_3sigma_range = np.quantile(new_samples, [0.00134989803163, 0.99865010196837])
            this_4sigma_range = np.quantile(new_samples, [0.000031671241833, 0.999968328758167])

            if not((0>this_1sigma_range[0])&(0<this_1sigma_range[1])):
                n_1sigma += 1

            if not((0>this_2sigma_range[0])&(0<this_2sigma_range[1])):
                n_2sigma += 1
            
            if not((0>this_3sigma_range[0])&(0<this_3sigma_range[1])):
                n_3sigma += 1
            
            if not((0>this_4sigma_range[0])&(0<this_4sigma_range[1])):
                n_4sigma += 1
            
            A_1sigma.append(this_1sigma_range)
            A_2sigma.append(this_2sigma_range)
            A_3sigma.append(this_3sigma_range)
            A_4sigma.append(this_4sigma_range)
            A_median.append(np.median(new_samples))
            A_average.append(np.average(new_samples))
            A_var.append(np.var(new_samples))

            #Compute the Credible Interval for the Amplitude
            credible = GetCDFThresholdPoint_1D_getdist(bare_posterior,norm, [0.682689, 0.954499, 0.997300203])

            A_cred_1sigma.append(credible[0])
            A_cred_2sigma.append(credible[1])
            A_cred_3sigma.append(credible[2])

    #End of the iterarion over all freq. ranges for a given mock. Saving the outputs
    out_f.create_dataset("n_1sigma", data = n_1sigma)
    out_f.create_dataset("n_2sigma", data = n_2sigma)
    out_f.create_dataset("n_3sigma", data = n_3sigma)
    out_f.create_dataset("n_4sigma", data = n_4sigma)
    out_f.create_dataset("1sigma", data = np.asarray(A_1sigma))
    out_f.create_dataset("2sigma", data = np.asarray(A_2sigma))
    out_f.create_dataset("3sigma", data = np.asarray(A_3sigma))
    out_f.create_dataset("4sigma", data = np.asarray(A_4sigma))
    out_f.create_dataset("omega_ctrs", data = np.asarray(omega_ctrs))
    out_f.create_dataset("omega_bin", data = omega_bin)
    out_f.create_dataset("A_median", data = np.asarray(A_median))
    out_f.create_dataset("A_average", data = np.asarray(A_average))
    out_f.create_dataset("A_cred_1sigma", data = np.asarray(A_cred_1sigma))
    out_f.create_dataset("A_cred_2sigma", data = np.asarray(A_cred_2sigma))
    out_f.create_dataset("A_cred_3sigma", data = np.asarray(A_cred_3sigma))
    out_f.create_dataset("A_var", data = np.asarray(A_var))
    out_f.close()

#TODO: Modify the output to be similar as the 1D case
def getJointCredibleInterval(posteriors, omega_min, omega_max, omega_bin, output_name,\
    range_handle="posterior:[{}, {}]", verbose = False):
    """
    input is a list of BinnedPosterior files
    """
    #Output file
    A_conf = []

    #Create a list to storage the files
    files_list = []

    for this_filename in posteriors:
        files_list.append(h5.File(results_directory+this_filename, 'r'))
    
    omega_bins = np.arange(omega_min, omega_max + omega_bin, omega_bin)
    omega_ctrs = 0.5*(omega_bins[1:] + omega_bins[:-1])

    #iterate over the frequency bins and get the results for each one separetely
    for i in tqdm(range(1, len(omega_bins))):
        this_key = range_handle.format(omega_bins[i-1], omega_bins[i])
        these_posteriors = []
        for this_file in files_list:
            try:
                these_posteriors.append(np.asarray(this_file[this_key]))
            except:
                these_posteriors.append(1)

        #Multiply the final posterior into a single one
        final_posterior = 1
        for x in these_posteriors:
            final_posterior *= x
        
        norm = np.sum(final_posterior*A_bin**2)
        final_posterior /= norm
        if verbose: print("Normalization factor: ", norm)

        #Get the credible interval
        A_conf.append(GetCDFThresholdPoint(final_posterior))
    
    output = np.vstack((omega_ctrs, np.array(A_conf)))
    np.savetxt(results_directory+output_name, output)

def getJointCredibleInterval_1D(posteriors, omega_min, omega_max, omega_bin, output_name,\
    range_handle="posterior:[{}, {}]", verbose = False):
    """
    input is a list of BinnedPosterior files
    """
    #File to storage the output
    try:
        out_f = h5.File(results_directory+output_name+".h5", 'w')
    except:
        print("Problem opening the file")
        sys.exit(-1)

    #Array to storage the credible intervals
    A_1sigma = []
    A_2sigma = []
    A_3sigma = []
    A_4sigma = []
    A_median = []
    A_average = []
    A_var = []

    #Array to storage the credible intervals for the amplitude
    A_cred_1sigma = []
    A_cred_2sigma = []
    A_cred_3sigma = []

    #count the number of events
    n_1sigma = 0
    n_2sigma = 0
    n_3sigma = 0
    n_4sigma = 0

    #Create a list to storage the files
    files_list = []

    for this_filename in posteriors:
        files_list.append(h5.File(results_directory+this_filename, 'r'))
    
    omega_bins = np.arange(omega_min, omega_max + omega_bin, omega_bin)
    omega_ctrs = 0.5*(omega_bins[1:] + omega_bins[:-1])

    #iterate over the frequency bins and get the results for each one separetely
    for i in tqdm(range(1, len(omega_bins))):
        this_key = range_handle.format(omega_bins[i-1], omega_bins[i])
        these_posteriors = []
        for this_file in files_list:
            try:
                these_posteriors.append(np.asarray(this_file[this_key][1]))
            except:
                these_posteriors.append(1)

        #Multiply the final posterior into a single one
        final_posterior = 1
        for x in these_posteriors:
            final_posterior *= x
        
        norm = np.sum(final_posterior*A_bin)
        final_posterior /= norm
        if verbose: print("Normalization factor: ", norm)

        #create the output file
        output_file = np.vstack((A_ctrs, final_posterior))

        #save this posterior in the output file
        out_f.create_dataset(this_key, data = output_file, \
            compression="gzip", compression_opts=9)    

        #Get new samples from the PDF to compute the statistics
        N = 100000 #number of samples to draw
        new_samples = np.random.choice(A_ctrs, size=N, p=final_posterior*A_bin, replace=True)

        #get the 1-sigma range
        this_1sigma_range = np.quantile(new_samples, [0.15865525393149998, 0.8413447460685])

        #get the 2-sigma range
        this_2sigma_range = np.quantile(new_samples, [0.022750131948179, 0.977249868051821])

        #get the 3-sigma range
        this_3sigma_range = np.quantile(new_samples, [0.00134989803163, 0.99865010196837])
        
        #get the 4-sigma range
        this_4sigma_range = np.quantile(new_samples, [0.000031671241833, 0.999968328758167])

        if not((0>this_1sigma_range[0])&(0<this_1sigma_range[1])):
            n_1sigma += 1

        if not((0>this_2sigma_range[0])&(0<this_2sigma_range[1])):
            n_2sigma += 1
        
        if not((0>this_3sigma_range[0])&(0<this_3sigma_range[1])):
            n_3sigma += 1
        
        if not((0>this_4sigma_range[0])&(0<this_4sigma_range[1])):
            n_4sigma += 1
        
        A_1sigma.append(this_1sigma_range)
        A_2sigma.append(this_2sigma_range)
        A_3sigma.append(this_3sigma_range)
        A_4sigma.append(this_4sigma_range)
        A_median.append(np.median(new_samples))
        A_average.append(np.mean(new_samples))
        A_var.append(np.var(new_samples))

        #Interp. Posterior
        final_posterior_interped = interp1d(A_ctrs, final_posterior)
        #Compute the Credible Interval for the Amplitude
        credible = GetCDFThresholdPoint_1D_getdist(final_posterior_interped, 1, [0.682, 0.954, 0.997])
        A_cred_1sigma.append(credible[0])
        A_cred_2sigma.append(credible[1])
        A_cred_3sigma.append(credible[2])

    #End of the iterarion over all freq. ranges for a given mock. Saving the outputs
    out_f.create_dataset("n_1sigma", data = n_1sigma)
    out_f.create_dataset("n_2sigma", data = n_2sigma)
    out_f.create_dataset("n_3sigma", data = n_3sigma)
    out_f.create_dataset("n_4sigma", data = n_4sigma)

    out_f.create_dataset("1sigma", data = np.asarray(A_1sigma))
    out_f.create_dataset("2sigma", data = np.asarray(A_2sigma))
    out_f.create_dataset("3sigma", data = np.asarray(A_3sigma))
    out_f.create_dataset("4sigma", data = np.asarray(A_4sigma))
    out_f.create_dataset("omega_ctrs", data = np.asarray(omega_ctrs))
    out_f.create_dataset("omega_bin", data = omega_bin)
    out_f.create_dataset("A_median", data = np.asarray(A_median))

    out_f.create_dataset("A_cred_1sigma", data = np.asarray(A_cred_1sigma))
    out_f.create_dataset("A_cred_2sigma", data = np.asarray(A_cred_2sigma))
    out_f.create_dataset("A_cred_3sigma", data = np.asarray(A_cred_3sigma))
    out_f.create_dataset("A_var", data = np.asarray(A_var))

    out_f.close()
    
#===================================================================================================
if __name__ == '__main__':
    pass