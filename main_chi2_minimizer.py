# main.py
import numpy as np
import ps_constructor
import likelihood
import mcmc_toolkit
import os
import argparse
import logging
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from multiprocessing import Pool
from dotenv import load_dotenv
import data_handling
import matplotlib.pyplot as plt
import os
from iminuit import Minuit

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
PROCESSES = int(os.getenv('PROCESSES'))

# Load the data products
k_file = os.getenv('DATA_k').format(args.mock)
DATA_NGC_file = os.getenv('DATA_NGC').format(args.mock)
DATA_SGC_file = os.getenv('DATA_SGC').format(args.mock)
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
DATA_NGC = data_handling.load_data(DATA_NGC_file, mask)
DATA_SGC = data_handling.load_data(DATA_SGC_file, mask)
DATA = np.concatenate((DATA_NGC, DATA_SGC))
covariance = data_handling.load_cov(COV_file, mask)
k = k[mask]
invcov = np.linalg.inv(covariance)

# Create the name of the data file
data_file_name = DATA_NGC_file.split('/')[-1].replace('.txt', '')
common_name = f"{prior_name}_omegamin_{OMEGA_MIN}_omegamax_{OMEGA_MAX}_{data_file_name}"
common_name = common_name.replace('desipipe_v4_2', "").replace("AbacusSummit", "").replace("z0.8-2.1", "").replace('desi_survey_catalogs_Y1', '').replace('mocks','')  # Remove Abacus and redshift references
common_name = common_name.replace('SecondGenMocks','').replace('__','').replace('IFFT_recsympk','recsym').replace('altmtl','').replace('NGC','').replace('SGC','')
common_name =  'MINUIT_' + common_name

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
logger.info(f'******************************************************** CHI2 minimizer ********************************************************')
logger.info(f'Processes: {PROCESSES}')
logger.info(f'DATA NGC file: {DATA_NGC_file}')
logger.info(f'DATA SGC file: {DATA_SGC_file}')
logger.info(f'COV file: {COV_file}')
logger.info(f'Window function (NGC): {fn_wf_ngc}')
logger.info(f'Window function (SGC): {fn_wf_sgc}')
logger.info(f'linear matter power spectrum: {PLIN}')
logger.info(f'primordial feature model: {primordialfeature_model}')
logger.info(f'prior_name: {prior_name}')
logger.info(f'priors_dir: {priors_dir}')
logger.info(f'KMIN: {KMIN}')
logger.info(f'KMAX: {KMAX}')
logger.info(f'OMEGA_MIN: {OMEGA_MIN}')
logger.info(f'OMEGA_MAX: {OMEGA_MAX}')
logger.info(f'Filename: {common_name}')

# Initialize the MCMC
mcmc = mcmc_toolkit.MCMC(1, prior_name, priors_dir=priors_dir, log_file='log/'+handle_log)
ndim_NGC = len(mcmc.input_prior['NGC'])
ndim_SGC = len(mcmc.input_prior['SGC'])
mcmc.set_walkers(5)
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
    theta_NGC = theta[0:ndim_NGC]
    theta_SGC = theta[ndim_NGC:ndim_NGC+ndim_SGC]
    shared_params = theta[ndim_NGC+ndim_SGC:]

    theta_NGC = theta_NGC + shared_params
    theta_SGC = theta_SGC + shared_params
    
    # Use np.concatenate to combine the results from both theories
    return np.concatenate((theory_NGC(theta_NGC), theory_SGC(theta_SGC)))
#***************************************************************************************************
#Create the likelihood
PrimordialFeature_likelihood = likelihood.likelihoods(theory, DATA, invcov)

def chi2(theta):
    return PrimordialFeature_likelihood.chi2(theta)

#Region in parameter to create the walkers ( Uniform[X0 +- DELTA] )
X0_str = os.getenv('X0')
DELTA_str = os.getenv('DELTA')

#Re-define the chains and figures directory
CHAIN_DIR = os.getenv('CHAIN_DIR')
if CHAIN_DIR:
    OUT_DIR = os.path.join(CHAIN_DIR,prior_name,f"omega_{int(OMEGA_MIN)}_{int(OMEGA_MAX)}",'minuit')
    os.makedirs(OUT_DIR, exist_ok=True)

if X0_str:
    X0 = np.array([float(x) for x in X0_str.split(',')])
    DELTA = np.array([float(x) for x in DELTA_str.split(',')])
else:
    X0 = np.array([])  # or handle the case where X0 is not set
    DELTA = np.array([])
    logger.warning('X0 and SIGMA not set')

def analyze_frequency(freq):
    """
    Analyze a single frequency and save the results to a separate file.
    """
    try:
        # Define parameter limits
        limits = [(mcmc.prior_bounds[0][i], mcmc.prior_bounds[1][i]) for i in range(mcmc.ndim)]
        limits[11] = (freq, freq)
        
        # Generate initial positions
        initial_positions = mcmc.create_walkers(initialize_walkers, x0=X0, delta=DELTA)
        initial_positions.T[11] = freq

        best_value = float('inf')
        best_params = None

        for this_guess in initial_positions:
            m = Minuit(chi2, this_guess)
            m.limits = limits
            m.migrad()

            if m.fval < best_value:
                best_value = m.fval
                best_params = m.values.to_dict()

        # Save results to a file
        result = {'best_value': best_value, 'best_params': best_params}
        output_filename = os.path.join(OUT_DIR, f"freq_{int(freq)}.npy")
        np.save(output_filename, result)

        logger.info(f"Frequency: {freq}, Best fval: {best_value}, Best params: {best_params}")
        logger.info(f"Results saved to {output_filename}")
        return freq, result
    
    except Exception as e:
        logger.error(f"Error analyzing frequency {freq}: {e}")
        return freq, None

if __name__ == '__main__':
    # Frequency range
    frequencies = np.arange(OMEGA_MIN + 5, OMEGA_MAX, 10)

    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Use multiprocessing Pool to process frequencies in parallel
    with Pool(processes = PROCESSES) as pool:
        results = list(tqdm(pool.imap(analyze_frequency, frequencies), total=len(frequencies)))

    # Optionally combine all results into a single dictionary (if needed)
    combined_results = {freq: res for freq, res in results if res is not None}

    # Save combined results as a backup
    combined_output_file = os.path.join(OUT_DIR, "combined_results.npy")
    np.save(combined_output_file, combined_results)
    logger.info(f"Combined results saved to {combined_output_file}")