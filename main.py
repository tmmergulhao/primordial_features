# main.py
import numpy as np
import ps_constructor
import likelihood
import mcmc_toolkit
import os
import argparse
import logging
import json
import sys
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from multiprocessing import Pool
from dotenv import load_dotenv
import data_handling
import matplotlib.pyplot as plt
import os
#########################################LOADING THE DATA###########################################
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run MCMC analysis with different setups.')
parser.add_argument('--env', type=str, required=True, help='Path to the .env file for the analysis setup')
parser.add_argument('--omega_min', type=float, required=True, help='Minimum value of omega')
parser.add_argument('--omega_max', type=float, required=True, help='Maximum value of omega')
parser.add_argument('--mock', type=int, required=False, help='What mock to use')
parser.add_argument('--handle', type=str, required=False, help='Add a prefix to the chains and log file')
args = parser.parse_args()

# The frequency range to be scanned
OMEGA_MIN = float(args.omega_min)
OMEGA_MAX = float(args.omega_max)

# Load environment variables from the specified .env file
load_dotenv(args.env)

# Whether or not to use multiprocessing
MULTIPROCESSING = os.getenv('MULTIPROCESSING')
PROCESSES = int(os.getenv('PROCESSES'))

# Load the data products
k_file = os.getenv('DATA_k')
DATA_file = os.getenv('DATA').format(args.mock)
COV_file = os.getenv('COV')
fn_wf_ngc = os.getenv('FN_WF_NGC')
fn_wf_sgc = os.getenv('FN_WF_SGC')

# Linear matter power spectrum (smooth and wiggly part)
PLIN = os.getenv('PLIN')

# Specify the primordial feature model
primordialfeature_model = os.getenv('MODEL')

# Get the prior
prior_name = os.getenv('PRIOR_NAME')
priors_dir = os.getenv('PRIORS_DIR')

# Number of walkers per free parameter
nwalkers_per_param = int(os.getenv('NWALKERS_PER_PARAM'))
initialize_walkers = os.getenv('INITIALIZE_WALKERS')

#Get the mask for the k-range
KMIN = float(os.getenv('KMIN')) if os.getenv('KMIN') is not None else None
KMAX = float(os.getenv('KMAX')) if os.getenv('KMAX') is not None else None

#########################################PREPARING THE DATA#########################################
# Load the gelman rubin convergence criteria
with open('gelman_rubin.json', 'r') as json_file:
            gelman_rubin = json.load(json_file)

# Load the window functions
if fn_wf_ngc is not None:
    wfunc_NGC = data_handling.load_winfunc(fn_wf_ngc)
    #Make sure the window function is normalised
    wfunc_NGC[1] = wfunc_NGC[1]/wfunc_NGC[1][0]

if fn_wf_sgc is not None:
    wfunc_SGC = data_handling.load_winfunc(fn_wf_sgc)
    #Make sure the window function is normalised
    wfunc_SGC[1] = wfunc_SGC[1]/wfunc_SGC[1][0]
    
# Load the k-array and apply the mask
k    = data_handling.load_data_k(k_file)
mask = data_handling.compute_mask(k, KMIN, KMAX)

# Load the filtered data and covariance
DATA = data_handling.load_data(DATA_file, mask)
covariance = data_handling.load_cov(COV_file, mask)
k = k[mask]
invcov = np.linalg.inv(covariance)

# Create the name of the data file
data_file_name = DATA_file.split('/')[-1].replace('.txt', '')
common_name = f"{prior_name}_omegamin_{OMEGA_MIN}_omegamax_{OMEGA_MAX}_kmin_{k[0]:.5f}_kmax_{k[-1]:.5f}_{data_file_name}"

if args.handle:
    handle_log = f"{args.handle}_{common_name}.log"
    handle = f"{args.handle}_{common_name}"
else:
    handle_log = f"{common_name}.log"
    handle = common_name

# Create the log file
logging.basicConfig(filename='log/'+handle, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log the variables
logger.info(f'Processes: {PROCESSES}')
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
logger.info(f'KMIN: {KMIN}')
logger.info(f'KMAX: {KMAX}')
logger.info(f'OMEGA_MIN: {OMEGA_MIN}')
logger.info(f'OMEGA_MAX: {OMEGA_MAX}')
logger.info(f'Filename: {common_name}')
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
    theory_NGC = lambda x: ps_model_NGC.Evaluate_bare(x)
    theory_SGC = lambda x: ps_model_SGC.Evaluate_bare(x)

else: #Convolve the theory with the window function
    ps_model_NGC.DefineWindowFunction(InterpolatedUnivariateSpline(wfunc_NGC[0],wfunc_NGC[1],ext=3))
    theory_NGC = lambda x: ps_model_NGC.Evaluate_wincov(x)

    ps_model_SGC.DefineWindowFunction(InterpolatedUnivariateSpline(wfunc_SGC[0],wfunc_SGC[1],ext=3))
    theory_SGC = lambda x: ps_model_SGC.Evaluate_wincov(x)

def theory(theta):
    # Slice theta to get the corresponding values for NGC and SGC
    theta_NGC = theta[[0] + list(range(2, 13))]
    theta_SGC = theta[[1] + list(range(2, 13))]
    
    # Use np.concatenate to combine the results from both theories
    return np.concatenate((theory_NGC(theta_NGC), theory_SGC(theta_SGC)))
#***************************************************************************************************
#Create the likelihood
PrimordialFeature_likelihood = likelihood.likelihoods(theory, DATA, invcov)

# Initialize the MCMC
mcmc = mcmc_toolkit.MCMC(1, prior_name, priors_dir=priors_dir, log_file='log/'+handle_log)
mcmc.set_walkers(nwalkers_per_param * mcmc.ndim)
mcmc.prior_bounds[0][11] = OMEGA_MIN
mcmc.prior_bounds[1][11] = OMEGA_MAX

in_prior_range = mcmc.in_prior

# Log the Gelman-Rubin convergence criteria
logger.info(f'Gelman-Rubin convergence criteria: {gelman_rubin}')
logger.info(f'Parameters: {mcmc.labels}')
logger.info(f'Range: {mcmc.prior_bounds}')
logger.info(f'MULTIPROCESSING: {MULTIPROCESSING}')

def logposterior(theta):
    if not in_prior_range(theta):
        return -np.inf
    else:
        return PrimordialFeature_likelihood.logGaussian(theta)

omega_ctr = 0.5*(mcmc.prior_bounds[0][11]+mcmc.prior_bounds[1][11])
omega_delta = 0.4*abs((mcmc.prior_bounds[0][11]-mcmc.prior_bounds[1][11]))

#Region in parameter to create the walkers ( Uniform[X0 +- DELTA] )
X0_str    = os.getenv('X0')
DELTA_str    = os.getenv('DELTA')

if X0_str:
    X0 = np.array([float(x) for x in X0_str.split(',')])
    DELTA = np.array([float(x) for x in DELTA_str.split(',')])
    X0[11] = omega_ctr
    DELTA[11] = omega_delta
else:
    X0 = np.array([])  # or handle the case where X0 is not set
    DELTA = np.array([])
    logger.warning('X0 and SIGMA not set')

#Re-define the chains and figures directory
CHAIN_DIR = os.getenv('CHAIN_DIR')
if CHAIN_DIR:
     mcmc.change_chain_dir(CHAIN_DIR) 

FIG_DIR = os.getenv('FIG_DIR')
if FIG_DIR:
    mcmc.change_fig_dir(FIG_DIR)

#Create the initial positions
initial_positions = [mcmc.create_walkers(initialize_walkers,x0 =X0,delta = DELTA) for _ in range(gelman_rubin['N'])]

if __name__ == '__main__':
    if MULTIPROCESSING:
        # Create a multiprocessing pool
        with Pool(processes = PROCESSES) as pool:
            #Run the MCMC simulation with Gelman-Rubin convergence criteria and multiprocessing pool
            mcmc.run(handle, 0, initial_positions, logposterior, pool=pool, 
            gelman_rubins=gelman_rubin)