# main.py
import numpy as np
import ps_constructor
import likelihood
import mcmc_toolkit
import pypower
import time
import os
import argparse
import logging
import json
import sys
from scipy.interpolate import interp1d
from multiprocessing import Pool
from dotenv import load_dotenv
import data_handling
import matplotlib.pyplot as plt

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
k_file = os.getenv('DATA_k')
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
    wfunc_NGC = data_handling.load_winfunc(fn_wf_ngc)

if fn_wf_sgc is not None:
    wfunc_SGC = data_handling.load_winfunc(fn_wf_sgc)

#Load the data
DATA = data_handling.load_data(DATA_file)
covariance = data_handling.load_cov(COV_file)
k = data_handling.load_data_k(k_file)

#********************** Defining the theory ********************************************************
# The data space has dimension 2*dim(k), since we jointly analyse NGC and SGC. Since the geomentry
# of NGC and SGC are different, they will have different window functions. It means that we will
# need to convolve the NGC and SGC separately. It can be done in the following way:
#       i)The total parameter space will have dimensions that are exclusive to a galaxy cap and
#        shared ones.
#        
#       ii)For a given input theta, we will split it into theta_NGC and theta SGC.
#
#       iii) Compute the theory for NGC and SGC
#
#       iv) Concatenate the result from the step above and compare with data.


# Initialize the model for NGC
ps_model_NGC = ps_constructor.PowerSpectrumConstructor(PLIN, primordialfeature_model, k)

# Initialize the model for SGC
ps_model_SGC = ps_constructor.PowerSpectrumConstructor(PLIN, primordialfeature_model, k)

if (fn_wf_ngc is None) or (fn_wf_sgc is None): #No window function convolution
    PrimordialFeature_theory_NGC = lambda x: ps_model_NGC.Evaluate_bare(x)
    PrimordialFeature_theory_SGC = lambda x: ps_model_SGC.Evaluate_bare(x)

else: #Convolve the theory with the window function
    ps_model_NGC.DefineWindowFunction(interp1d(wfunc_NGC[0],wfunc_NGC[1]))
    theory_NGC = lambda x: ps_model_NGC.Evaluate_wincov(x)

    ps_model_SGC.DefineWindowFunction(interp1d(wfunc_SGC[0],wfunc_SGC[1]))
    theory_SGC = lambda x: ps_model_SGC.Evaluate_wincov(x)

#theta = ["BNGC", "BSGC", "sigma_nl", "sigma_s", "a0", "a1", "a2", "a3", "a4", "alpha", "A_lin", "omega_lin", "phi"]
#Create the theory for the whole data space
def theory(theta):
    theta_NGC = [theta[0],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7],theta[8],theta[9],theta[10],theta[11],theta[12]]
    theta_SGC = [theta[1],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7],theta[8],theta[9],theta[10],theta[11],theta[12]]
    return np.hstack((theory_NGC(theta_NGC),theory_SGC(theta_SGC)))

#***************************************************************************************************
invcov = np.linalg.inv(covariance)

#Create the likelihood
PrimordialFeature_likelihood = likelihood.likelihoods(theory, DATA, invcov)

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

print(logposterior(initial_positions[0][0]))
sys.exit(-1)
if __name__ == '__main__':
    if MULTIPROCESSING:
        # Create a multiprocessing pool
        with Pool(processes = int(PROCESSES)) as pool:
            #Run the MCMC simulation with Gelman-Rubin convergence criteria and multiprocessing pool
            mcmc.run(handle, 0, initial_positions, logposterior, pool=pool, 
            gelman_rubins=gelman_rubin)