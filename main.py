# main.py

import numpy as np
import ps_constructor
import likelihood
import mcmc_toolkit
import pypower
from multiprocessing import Pool
import time
from dotenv import load_dotenv
import os
import argparse
import logging
import json
import sys
from scipy.interpolate import interp1d

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run MCMC analysis with different setups.')
parser.add_argument('--env', type=str, required=True, help='Path to the .env file for the analysis setup')
args = parser.parse_args()

# Load environment variables from the specified .env file
load_dotenv(args.env)

#Wheter or not to use multiprocessing
MULTIPROCESSING = os.getenv('MULTIPROCESSING')
PROCESSES = os.getenv('PROCESSES')

# Load the data products
DATA_file = os.getenv('DATA')
COV_file = os.getenv('COV')
fn_wf_ngc = os.getenv('FN_WF_NGC')
fn_wf_sgc = os.getenv('FN_WF_SGC')

# linear matter power spectrum (smooth and wiggly part)
PLIN = os.getenv('PLIN')

#Specify the primordial feature model
primordialfeature_model = os.getenv('MODEL')

#Get the prior
prior_name = os.getenv('PRIOR_NAME')
priors_dir = os.getenv('PRIORS_DIR')

#number of walkers per free parameter
nwalkers_per_param = int(os.getenv('NWALKERS_PER_PARAM'))

# Configure logging
handle = '_'.join([prior_name, DATA_file.split('/')[-1].split('.npy')[0]])+'.log'

logging.basicConfig(filename='log/'+handle, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

#Load the gelman rubin convergence criteria
with open('gelman_rubin.json', 'r') as json_file:
            gelman_rubin = json.load(json_file)

# Log the variables
logger.info(f'DATA file: {DATA_file}')
logger.info(f'COV file: {COV_file}')
logger.info(f'Window function (NGC): {fn_wf_ngc}')
logger.info(f'Window function (SGC): {fn_wf_sgc}')
logger.info(f'linear matter power spectrum: {PLIN}')
logger.info(f'primordial feature model: {primordialfeature_model}')
logger.info(f'prior_name: {prior_name}')
logger.info(f'nwalkers_per_param: {nwalkers_per_param}')
logger.info(f'priors_dir: {priors_dir}')
logger.info(f'MULTIPROCESSING: {MULTIPROCESSING}')

if MULTIPROCESSING:
    logger.info(f'Processes: {PROCESSES}')

if fn_wf_ngc is not None:
    wfunc_NGC = np.loadtxt(fn_wf_ngc)

if fn_wf_sgc is not None:
    wfunc_SGC = np.loadtxt(fn_wf_sgc)

#Load the data
poles = pypower.PowerSpectrumStatistics.load(DATA_file)
covariance = np.loadtxt(COV_file)
invcov = np.linalg.inv(covariance)

# Unpack the power spectrum
k, DATA = poles.k, poles(ell=0).real

# Initialize the model
ps_model = ps_constructor.PowerSpectrumConstructor(PLIN, primordialfeature_model, k)

if (fn_wf_ngc is  None) or (fn_wf_sgc is  None):
    PrimordialFeature_theory = lambda x: ps_model.Evaluate_bare(x)# NGC and SGC are concatenated

else:

    ps_model.DefineWindowFunction(interp1d(wfunc_NGC[0],wfunc_NGC[1]),
                                    interp1d(wfunc_SGC[0],wfunc_SGC[1]))
    PrimordialFeature_theory = lambda x: ps_model.Evaluate_wincov(x)# NGC and SGC are concatenated

#Create the likelihood
PrimordialFeature_likelihood = likelihood.likelihoods(PrimordialFeature_theory, DATA, invcov)

# Initialize the MCMC
mcmc = mcmc_toolkit.MCMC(1, prior_name, priors_dir=priors_dir, log_file='log/'+handle)
mcmc.set_walkers(nwalkers_per_param * mcmc.ndim)

in_prior_range = mcmc.in_prior

# Log the Gelman-Rubin convergence criteria
logger.info(f'Gelman-Rubin convergence criteria: {gelman_rubin}')

def logposterior(theta):
    if not in_prior_range(theta):
        return -np.inf
    else:
        return PrimordialFeature_likelihood.logGaussian(theta)


initial_positions = [mcmc.create_walkers('uniform_prior') for _ in range(gelman_rubin['N'])]

if __name__ == '__main__':
    if MULTIPROCESSING:
        # Create a multiprocessing pool
        with Pool(processes = int(PROCESSES)) as pool:
            #Run the MCMC simulation with Gelman-Rubin convergence criteria and multiprocessing pool
            mcmc.run(handle, 0, initial_positions, logposterior, pool=pool, 
            gelman_rubins=gelman_rubin)