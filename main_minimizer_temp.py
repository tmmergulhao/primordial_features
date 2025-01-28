import numpy as np
import matplotlib.pyplot as plt
import sys, os
import json
import ps_constructor, likelihood, mcmc_toolkit, data_handling
import logging
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm
from iminuit import Minuit
from dotenv import load_dotenv
import multiprocessing
from functools import partial

#SET UP THE PATHS
MACHINE = 'MAC'
PATHS = data_handling.load_json_to_dict('paths.json')[MACHINE]

#ENVIRONMENT NAME
envname = "envs/DATA/Y1/lin/DESI_LRG1_DATA_prerecon.env"

#Load the environment and paths
load_dotenv(os.path.join(PATHS['MAIN_DIR'],envname))
MAIN_DIR = PATHS['MAIN_DIR']
DATA_DIR = PATHS['DATA_DIR']
FIG_DIR = PATHS['FIG_DIR']
CHAIN_DIR = PATHS['CHAIN_DIR']


prior_name = os.getenv('PRIOR_NAME')
prior_file = os.path.join(MAIN_DIR, 'priors', prior_name)

CHAIN_FOLDER = os.getenv('CHAIN_FOLDER')
CHAIN_PATH = os.path.join(CHAIN_DIR, CHAIN_FOLDER, prior_name,'MINUIT')

FIG_FOLDER = os.getenv('FIG_FOLDER')
FIG_PATH = os.path.join(FIG_DIR, FIG_FOLDER, prior_name)
DATA_FLAG = True

DATA_NGC_file = os.getenv('DATA_NGC')
DATA_NGC_file = os.path.join(DATA_DIR, DATA_NGC_file)

DATA_SGC_file = os.getenv('DATA_SGC')
DATA_SGC_file = os.path.join(DATA_DIR, DATA_SGC_file)

COV_NGC_file = os.getenv('COV_NGC')
COV_NGC_file = os.path.join(DATA_DIR, COV_NGC_file)

COV_SGC_file = os.getenv('COV_SGC')
COV_SGC_file = os.path.join(DATA_DIR, COV_SGC_file)

fn_wf_ngc = os.getenv('FN_WF_NGC')
if fn_wf_ngc is not None:
    fn_wf_ngc = os.path.join(DATA_DIR, fn_wf_ngc)

fn_wf_sgc = os.getenv('FN_WF_SGC')
if fn_wf_sgc is not None:
    fn_wf_sgc = os.path.join(DATA_DIR, fn_wf_sgc)

# Linear matter power spectrum (smooth and wiggly part)
PLIN = os.getenv('PLIN')
PLIN = os.path.join(MAIN_DIR, PLIN)

# Specify the primordial feature model
primordialfeature_model = os.getenv('MODEL')

# Number of walkers per free parameter
nwalkers_per_param = int(os.getenv('NWALKERS_PER_PARAM'))
initialize_walkers = os.getenv('INITIALIZE_WALKERS')

#Get the mask for the k-range
KMIN = float(os.getenv('KMIN')) if os.getenv('KMIN') is not None else None
KMAX = float(os.getenv('KMAX')) if os.getenv('KMAX') is not None else None

# Create the name of the data file
data_label = envname.split('/')[-1].split('.')[0]

common_name = f"DATA_{data_label}_{prior_name}"

handle = common_name

# Load the k-array and apply the mask
data_processor = data_handling.DataProcessor(KMIN, KMAX)
k,DATA_NGC = data_processor.load_data(DATA_NGC_file)
k,DATA_SGC = data_processor.load_data(DATA_SGC_file)
DATA = np.concatenate((DATA_NGC, DATA_SGC))

COV_NGC = data_processor.load_cov(COV_NGC_file)
COV_SGC = data_processor.load_cov(COV_SGC_file)
COV = np.block([[COV_NGC, np.zeros_like(COV_NGC)], [np.zeros_like(COV_SGC), COV_SGC]])
invCOV = np.linalg.inv(COV)

#HARTLAP CORRECTION
Nmocks = 1000
Nb = len(k)
invCOV *= (Nmocks-Nb-2-1)/(Nmocks-1)

# Load the window functions
if fn_wf_ngc is not None:
    wfunc_NGC = data_handling.load_winfunc(fn_wf_ngc)
    #Make sure the window function is normalised
    wfunc_NGC[1] = wfunc_NGC[1]/wfunc_NGC[1][0]

if fn_wf_sgc is not None:
    wfunc_SGC = data_handling.load_winfunc(fn_wf_sgc)
    #Make sure the window function is normalised
    wfunc_SGC[1] = wfunc_SGC[1]/wfunc_SGC[1][0]

# Create the log file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log the variables
logger.info(f'DATA NGC file: {DATA_NGC_file}')
logger.info(f'DATA SGC file: {DATA_SGC_file}')
logger.info(f'COV NGC file: {COV_NGC_file}')
logger.info(f'Window function (NGC): {fn_wf_ngc}')
logger.info(f'Window function (SGC): {fn_wf_sgc}')
logger.info(f'linear matter power spectrum: {PLIN}')
logger.info(f'primordial feature model: {primordialfeature_model}')
logger.info(f'prior_name: {prior_name}')
logger.info(f'nwalkers_per_param: {nwalkers_per_param}')
logger.info(f'KMIN: {KMIN}')
logger.info(f'KMAX: {KMAX}')
logger.info(f'Filename: {common_name}')

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

def theory(*theta):
    # Slice theta to get the corresponding values for NGC and SGC
    theta=theta[0]
    
    theta_NGC = theta[0:ndim_NGC]
    theta_SGC = theta[ndim_NGC:ndim_NGC+ndim_SGC]
    shared_params = theta[ndim_NGC+ndim_SGC:]
    
    theta_NGC = np.concatenate([theta_NGC, shared_params])
    theta_SGC = np.concatenate([theta_SGC, shared_params])
    
    # Use np.concatenate to combine the results from both theories
    return np.concatenate((theory_NGC(theta_NGC), theory_SGC(theta_SGC)))

# Load the gelman rubin convergence criteria
with open(os.path.join(MAIN_DIR,'gelman_rubin.json'), 'r') as json_file:
            gelman_rubin = json.load(json_file)

# Initialize the MCMC
mcmc = mcmc_toolkit.MCMC(1, prior_file)
ndim_NGC = len(mcmc.input_prior['NGC'])
ndim_SGC = len(mcmc.input_prior['SGC'])
mcmc.gelman_rubin(gelman_rubin)
mcmc.set_walkers(1)

PrimordialFeature_likelihood = likelihood.likelihoods(theory, DATA, invCOV)

def chi2(*theta):
    return PrimordialFeature_likelihood.chi2(list(theta))

omegas = np.arange(105,4005,10)
limits = [(x[0],x[1]) for x in mcmc.prior_bounds.T]

X0_str = os.getenv("X0")
DELTA_str = os.getenv('DELTA')

if X0_str:
    X0 = np.array([float(x) for x in X0_str.split(',')])
    DELTA = np.array([float(x) for x in DELTA_str.split(',')])
else:
    X0 = np.array([])  # or handle the case where X0 is not set
    DELTA = np.array([])
    logger.warning('X0 and SIGMA not set')

# Function to compute chi2 for a given omega and save to a file
def compute_and_save_chi2(omega, mcmc, X0, DELTA, chi2_func, limits):
    """
    Computes chi2 for a given omega, runs optimization multiple times with different initial guesses,
    and saves the best result to a file.

    Parameters:
    - omega: The omega value for the current iteration.
    - mcmc: MCMC object containing setup and priors.
    - X0: Initial walker positions.
    - DELTA: Step size for initial walker positions.
    - chi2_func: The chi2 function.
    - limits: Parameter limits.
    """
    # Update the omega limit
    limits[mcmc.id_map['omega']] = (omega, omega)

    # Create initial positions
    mcmc.set_walkers(10)
    initial_positions = mcmc.create_walkers(initialize_walkers, x0=X0, delta=DELTA)

    best_chi2 = float('inf')  # Initialize with infinity
    best_params = None

    for i in range(10):
        # Get the current initial position
        init_pos = initial_positions[i]
        init_pos[mcmc.id_map['omega']] = omega

        # Run the optimization
        m = Minuit(chi2_func, name=mcmc.labels, **{x: val for x, val in zip(mcmc.labels, init_pos)})
        m.limits = limits
        m.migrad(ncall=20000)

        # Check if this is the best result so far
        if m.fmin.fval < best_chi2:
            best_chi2 = m.fmin.fval
            best_params = list(m.values.to_dict().values())

    # Save the best result to a file
    filename = CHAIN_PATH + f'/chi2_{omega}_{omega}.txt'
    with open(filename, 'w') as f:
        f.write(f'chi2: {best_chi2}\n')
        f.write(f'params: {best_params}\n')

    print(f"Processed omega: {omega} -> Saved best result to {filename}")

# Parallel execution setup
if __name__ == '__main__':
    # List of omegas to iterate over
    omegas = np.arange(105, 115, 10)

    # Partial function to pass static arguments
    compute_func = partial(compute_and_save_chi2, mcmc=mcmc, X0=X0, DELTA=DELTA, chi2_func=chi2, limits=limits)

    # Use multiprocessing Pool for parallel processing
    with multiprocessing.Pool(processes=1) as pool:
        pool.map(compute_func, omegas)