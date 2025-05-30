import numpy as np
import ps_constructor
import likelihood
import mcmc_toolkit
import argparse
import logging
import sys,os
from scipy.interpolate import InterpolatedUnivariateSpline
from multiprocessing import Pool
from dotenv import load_dotenv
import data_handling
from plot_results import *
import postprocessing as pp

#########################################LOADING THE DATA###########################################
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run MCMC analysis with different setups.')
parser.add_argument('--data_env', type=str, required=True, help='Path to the .env file for the data')
parser.add_argument('--sampler_env', type=str, required=True, help='Path to the .env file for the sampling setup')
parser.add_argument('--mock', type=int, required=True, help='What mock to use. Set -1 for data, and 0,1,2,... for mocks')
parser.add_argument('--machine', type =str,required=True)
parser.add_argument('--reconstruction',default=True, type=str, required=True, help='Whether to use the reconstructed power spectrum')
parser.add_argument('--omega_min', type=float, required=False, help='Minimum value of omega')
parser.add_argument('--omega_max', type=float, required=False, help='Maximum value of omega')
parser.add_argument('--handle', type=str, required=False, help='Add a prefix to the chains and log file')
parser.add_argument('--processess', type=int, required=False, help='Number of processes to use')
parser.add_argument('--debug',default=False, type=bool, required=False, help='Number of processes to use')
parser.add_argument('--postprocess',type=str, required=False, help='Post process the chains')
parser.add_argument('--run',default=True, type=str, required=False, help='Run the chain')
parser.add_argument('--EZMOCK',default=False, type=str, required=False, help='Whether to use EZMocks')
parser.add_argument('--sampler', default='emcee', type=str, required=False, help='Sampler to use')
args = parser.parse_args()

#Load the paths
MACHINE = args.machine
SAMPLER = args.sampler.lower()
PATHS = data_handling.load_json_to_dict('paths.json')[MACHINE]
MAIN_DIR = PATHS['MAIN_DIR']
DATA_DIR = PATHS['DATA_DIR']
FIG_DIR = PATHS['FIG_DIR']
CHAIN_DIR = PATHS['CHAIN_DIR']

# Load the environment variables
load_dotenv(args.data_env)
load_dotenv(args.sampler_env)

# The frequency range to be scanned
if args.omega_min and args.omega_max:
    FREQS = True
    OMEGA_MIN = float(args.omega_min)
    OMEGA_MAX = float(args.omega_max)
else:
    FREQS = False

if SAMPLER not in ['emcee', 'pocomc']:
    raise ValueError(f"Invalid sampler: {SAMPLER}. Choose 'emcee' or 'pocomc'.")

################################ Reading the analysis Specs ########################################
if args.mock <0:
    DATA_FLAG = True
    suffix = "DATA"
else:
    DATA_FLAG = False
    suffix = "MOCK"

reconstruction_flag = str(args.reconstruction).lower() in ['true', '1', 'yes']
run_chain_flag = str(args.run).lower() in ['true', '1', 'yes']
EZMOCK_flag = str(args.EZMOCK).lower() in ['true', '1', 'yes']
debug_flag = str(args.debug).lower() in ['true', '1', 'yes']
postprocess_flag = str(args.postprocess).lower() in ['true', '1', 'yes']

data_mode = "POST" if reconstruction_flag else "PRE"

# Construct the keys for NGC and SGC for both data and covariance files
if EZMOCK_flag:
    data_ngc_key = f"PK_NGC_{data_mode}_EZ{suffix}"
    data_sgc_key = f"PK_SGC_{data_mode}_EZ{suffix}"
else:
    data_ngc_key = f"PK_NGC_{data_mode}_{suffix}"
    data_sgc_key = f"PK_SGC_{data_mode}_{suffix}"
    
cov_ngc_key  = f"COV_NGC_{data_mode}"
cov_sgc_key  = f"COV_SGC_{data_mode}"

# Retrieve the paths from the environment
DATA_NGC_file = os.getenv(data_ngc_key).format(args.mock)
DATA_SGC_file = os.getenv(data_sgc_key).format(args.mock)
COV_NGC_file  = os.getenv(cov_ngc_key)
COV_SGC_file  = os.getenv(cov_sgc_key)
DATA_NGC_file = os.path.join(DATA_DIR, DATA_NGC_file)
DATA_SGC_file = os.path.join(DATA_DIR, DATA_SGC_file)
COV_NGC_file = os.path.join(DATA_DIR, COV_NGC_file)
COV_SGC_file = os.path.join(DATA_DIR, COV_SGC_file)

# Get the prior
prior_name = os.getenv('PRIOR_NAME')
prior_file = os.path.join(MAIN_DIR, 'priors', prior_name)

#Set the chains folder
OUT_FOLDER = os.getenv('OUT_FOLDER')
if DATA_FLAG:
    OUT_FOLDER = os.path.join(OUT_FOLDER, 'DATA')
else:
    OUT_FOLDER = os.path.join(OUT_FOLDER, f'MOCK_{args.mock}')

if FREQS:
    CHAIN_PATH = os.path.join(CHAIN_DIR, OUT_FOLDER, prior_name, f'{OMEGA_MIN}_{OMEGA_MAX}')
else:
    CHAIN_PATH = os.path.join(CHAIN_DIR, OUT_FOLDER, prior_name)

#Set the Figures folder
if FREQS:
    FIG_PATH = os.path.join(FIG_DIR, OUT_FOLDER, prior_name, f'{OMEGA_MIN}_{OMEGA_MAX}')
else:
    FIG_PATH = os.path.join(FIG_DIR, OUT_FOLDER, prior_name)

# Whether or not to use multiprocessing
MULTIPROCESSING = os.getenv('MULTIPROCESSING')
PROCESSES = int(os.getenv('PROCESSES'))

if args.processess:
    PROCESSES = args.processess

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

GELMAN_N = int(os.getenv('GELMAN_N'))
GELMAN_EPS = float(os.getenv('GELMAN_EPS'))
GELMAN_MIN = int(os.getenv('GELMAN_MIN'))
GELMAN_CONV_STEPS = int(os.getenv('GELMAN_CONV_STEPS'))
gelman_rubin = {
        "N":GELMAN_N,
        "epsilon":GELMAN_EPS,
        "min_length":GELMAN_MIN,
        "convergence_steps":GELMAN_CONV_STEPS
    }
    
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
data_processor = data_handling.DataProcessor(KMIN, KMAX)

#adjust the skip value depending on the file
if 'boss' not in DATA_NGC_file.lower():
    k,DATA_NGC = data_processor.load_data_DESI(DATA_NGC_file)
    k,DATA_SGC = data_processor.load_data_DESI(DATA_SGC_file)
    DATA = np.concatenate((DATA_NGC, DATA_SGC))
else:
    k,DATA_NGC = data_processor.load_data_BOSS(DATA_NGC_file)
    k,DATA_SGC = data_processor.load_data_BOSS(DATA_SGC_file)
    DATA = np.concatenate((DATA_NGC, DATA_SGC))

COV_NGC = data_processor.load_cov(COV_NGC_file)
COV_SGC = data_processor.load_cov(COV_SGC_file)
COV = np.block([[COV_NGC, np.zeros_like(COV_NGC)], [np.zeros_like(COV_SGC), COV_SGC]])
invCOV = np.linalg.inv(COV)

#HARTLAP CORRECTION
Nmocks = 1000
Nb = len(k)
invCOV *= (Nmocks-Nb-2)/(Nmocks-1)

# Create the name of the data file
data_label = args.data_env.split('/')[-1].split('.')[0]

if DATA_FLAG:
    prefix = "DATA"
else:
    prefix = f"MOCK_{args.mock}"

suffix = f"_{OMEGA_MIN}_{OMEGA_MAX}" if FREQS else ""
common_name = f"{prefix}_{data_mode}_{data_label}_{prior_name}{suffix}"

if args.handle:
    handle_log = f"{args.handle}_{common_name}.log"
    handle = f"{args.handle}_{common_name}"
else:
    handle_log = f"{common_name}.log"
    handle = common_name

# Construct the log file path inside the chains directory
log_filename = os.path.join(CHAIN_PATH, f"{handle}.log")
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

# Configure logging based on how the file is executed
if __name__ == '__main__':
    
    # When imported, log to a file
    log_filename = log_filename
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

else:
    # When running directly, log to stdout (e.g., in a Jupyter Notebook or terminal)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


logger = logging.getLogger(__name__)
# Log the variables
logger.info(f'Processes: {PROCESSES}')
logger.info(f'DATA NGC file: {DATA_NGC_file}')
logger.info(f'DATA SGC file: {DATA_SGC_file}')
logger.info(f'COV NGC file: {COV_NGC_file}')
logger.info(f'COV SGC file: {COV_SGC_file}')
logger.info(f'Window function (NGC): {fn_wf_ngc}')
logger.info(f'Window function (SGC): {fn_wf_sgc}')
logger.info(f'linear matter power spectrum: {PLIN}')
logger.info(f'primordial feature model: {primordialfeature_model}')
logger.info(f'prior_name: {prior_name}')
logger.info(f'nwalkers_per_param: {nwalkers_per_param}')
logger.info(f'MULTIPROCESSING: {MULTIPROCESSING}')
logger.info(f'KMIN: {KMIN}')
logger.info(f'KMAX: {KMAX}')
logger.info(f'Filename: {common_name}')

if primordialfeature_model != 'None':
    logger.info(f'OMEGA_MIN: {OMEGA_MIN}')
    logger.info(f'OMEGA_MAX: {OMEGA_MAX}')

# Initialize the MCMC
mcmc = mcmc_toolkit.MCMC(1, prior_file, log_file='log/'+handle_log)
ndim_NGC = len(mcmc.input_prior['NGC'])
ndim_SGC = len(mcmc.input_prior['SGC'])

#Gelman Rubin convergence criteria
mcmc.set_gelman_rubin(gelman_rubin)

if debug_flag:
    mcmc.set_gelman_rubin({
        "N":1,
        "epsilon":10,
        "min_length":5000,
        "convergence_steps":10
    })
    fn_wf_ngc = None
    fn_wf_sgc = None
    MULTIPROCESSING = False
#********************** Defining the theory ********************************************************
# Initialize the model for NGC
ps_model_NGC = ps_constructor.PowerSpectrumConstructor(k, ps_filename=PLIN, pf_model=primordialfeature_model)

# Initialize the model for SGC
ps_model_SGC = ps_constructor.PowerSpectrumConstructor(k, ps_filename=PLIN, pf_model = primordialfeature_model)

if (fn_wf_ngc is None) or (fn_wf_sgc is None): #No window function convolution
    theory_NGC = lambda x: ps_model_NGC.Evaluate_bare(x)
    theory_SGC = lambda x: ps_model_SGC.Evaluate_bare(x)

else: #Convolve the theory with the window function
    ps_model_NGC.DefineWindowFunction(InterpolatedUnivariateSpline(wfunc_NGC[0],wfunc_NGC[1],ext=3))
    theory_NGC = lambda x: ps_model_NGC.Evaluate_winconv(x)

    ps_model_SGC.DefineWindowFunction(InterpolatedUnivariateSpline(wfunc_SGC[0],wfunc_SGC[1],ext=3))
    theory_SGC = lambda x: ps_model_SGC.Evaluate_winconv(x)

def theory(theta):
    # Slice theta to get the corresponding values for NGC and SGC
    theta_NGC = theta[0:ndim_NGC]
    theta_SGC = theta[ndim_NGC:ndim_NGC+ndim_SGC]
    shared_params = theta[ndim_NGC+ndim_SGC:]

    theta_NGC = np.concatenate([theta_NGC, shared_params])
    theta_SGC = np.concatenate([theta_SGC, shared_params])
    
    # Use np.concatenate to combine the results from both theories
    return np.concatenate((theory_NGC(theta_NGC), theory_SGC(theta_SGC)))

#***************************************************************************************************
#Create the likelihood
PrimordialFeature_likelihood = likelihood.likelihoods(theory, DATA, invCOV)

# Initialize the MCMC
mcmc.set_walkers(nwalkers_per_param * mcmc.ndim)

if primordialfeature_model != 'None':
    mcmc.prior_bounds[0][mcmc.id_map['omega']] = OMEGA_MIN
    mcmc.prior_bounds[1][mcmc.id_map['omega']] = OMEGA_MAX
    
in_prior_range = mcmc.in_prior

# Log the Gelman-Rubin convergence criteria
logger.info(f'Gelman-Rubin convergence criteria: {gelman_rubin}')
logger.info(f'MULTIPROCESSING: {MULTIPROCESSING}')

def logposterior(theta):
    if not in_prior_range(theta):
        return -np.inf
    else:
        return PrimordialFeature_likelihood.logGaussian(theta)
    
def logposterior_no_prior(theta):
    return PrimordialFeature_likelihood.logGaussian(theta)

if primordialfeature_model != 'None':
    omega_ctr = 0.5*(mcmc.prior_bounds[0][mcmc.id_map['omega']]+mcmc.prior_bounds[1][mcmc.id_map['omega']])
    omega_delta = 0.4*abs((mcmc.prior_bounds[0][11]-mcmc.prior_bounds[1][11]))

#Region in parameter to create the walkers ( Uniform[X0 +- DELTA] )
X0_str    = os.getenv('X0')
DELTA_str    = os.getenv('DELTA')

if X0_str:
    X0 = np.array([float(x) for x in X0_str.split(',')])
    DELTA = np.array([float(x) for x in DELTA_str.split(',')])

    if primordialfeature_model != 'None':
        X0[11] = omega_ctr
        DELTA[11] = omega_delta
else:
    X0 = np.array([])  # or handle the case where X0 is not set
    DELTA = np.array([])
    logger.warning('X0 and SIGMA not set')

#Re-define the chains and figures directory
os.makedirs(CHAIN_PATH, exist_ok=True)
mcmc.change_chain_dir(CHAIN_PATH) 

os.makedirs(FIG_PATH, exist_ok=True)
mcmc.change_fig_dir(FIG_PATH)

#Create the initial positions
initial_positions = [mcmc.create_walkers(initialize_walkers,x0 =X0,delta = DELTA) for _ in range(mcmc.gelman_rubin['N'])]

if __name__ == '__main__':
    if SAMPLER == 'emcee':
        if run_chain_flag:
            if MULTIPROCESSING:
                # Create a multiprocessing pool
                with Pool(processes = PROCESSES) as pool:
                    #Run the MCMC simulation with Gelman-Rubin convergence criteria and multiprocessing pool
                    mcmc.run(handle, 1, initial_positions, logposterior, pool=pool, 
                    gelman_rubin=True, new=True, plots=True)
            else:
                #Run the MCMC simulation with Gelman-Rubin convergence criteria
                mcmc.run(handle, 1, initial_positions, logposterior, gelman_rubin=True, new=True, plots=True)
                
    if SAMPLER == 'pocomc':
        import pocomc as pc
        from scipy.stats import uniform

        loc = mcmc.prior_bounds[0]
        scale = mcmc.prior_bounds[1] - mcmc.prior_bounds[0]
        prior = pc.Prior([uniform(loc[i], scale[i]) for i in range(len(loc))])
        
        if MULTIPROCESSING:
            with Pool(processes = PROCESSES) as pool:
                sampler = pc.Sampler(
                prior=prior,
                likelihood=logposterior_no_prior,
                vectorize=False,
                random_state=0,
                n_effective = 800,
                n_active = None,
                output_dir = CHAIN_PATH,
                output_label = handle,
                pool = pool
                )  
                if run_chain_flag:
                    sampler.run(save_every=10,n_total= 150_000)
        else:
            sampler = pc.Sampler(
                prior=prior,
                likelihood=logposterior_no_prior,
                vectorize=False,
                random_state=0,
                n_effective = 800,
                n_active = None,
                output_dir = CHAIN_PATH,
                output_label = handle
            )  
            if run_chain_flag:
                sampler.run(save_every=10,n_total = 150_000)
        #sampler.load_state('/cosma8/data/dp322/dc-merg1/chains/BOSS_highz/DATA/CPSC_singlepol/13.7_20.0/new_sample_DATA_POST_BOSS_highz_CPSC_singlepol_13.7_20.0_450.state')
        logger.info(f"Sampler finished. Output saved to {CHAIN_PATH}")
        samples, weights, logl, logp = sampler.posterior()
        np.save(os.path.join(CHAIN_PATH,'new_sample.npy'), samples)
        np.save(os.path.join(CHAIN_PATH,'new_weights.npy'), weights)
        np.save(os.path.join(CHAIN_PATH,'new_logl.npy'), logl)
        np.save(os.path.join(CHAIN_PATH,'new_logp.npy'), logp)
        logz, logz_err = sampler.evidence()
        logger.info(f"logZ: {logz} ± {logz_err}")
        import corner
        import matplotlib.pyplot as plt
        fig = corner.corner(samples, weights=weights, color="C0")
        plt.savefig(os.path.join(CHAIN_PATH,'new_corner.png'))

# After MCMC chains converge
    try:
        plot_results(
            mcmc=mcmc,
            likelihood=PrimordialFeature_likelihood,
            theory=theory,
            DATA=DATA,
            COV_NGC=COV_NGC,
            COV_SGC=COV_SGC,
            k=k,
            FIG_PATH=FIG_PATH,
            handle=handle,
            primordialfeature_model = primordialfeature_model,
            save_chi2=True)
    except: 
        pass
    
    if postprocess_flag:
        FREQ_BIN = int(os.getenv('FREQ_BIN'))
        BINNING_AXIS = mcmc.id_map['omega']

        # Directories and filenames
        #f_bare = mcmc.chain_file_paths[0].split('_Run_')[0]
        f_bare = mcmc.generate_chain_file_paths(handle,gelman_rubin)[0].split('_Run_')[0]
        freqs = [[OMEGA_MIN, OMEGA_MAX]]

        # Number of chains to load
        n = mcmc.gelman_rubin.get('N')

        #File paths
        f_total_chain = f_bare + '_combined.h5'
        f_binned_chain = f_bare + '_binned.h5'
        f_analysis = f_bare + '_analysis.h5'
        
        print(f_total_chain)
        print(f_binned_chain)
        print(f_analysis)

        # Process chains
        pp.get_total_chain(f_bare, f_total_chain, n, burnin_frac=0.3, thin=10)
        pp.BinnedChain([f_total_chain], 
                       freqs, 
                       f_binned_chain, 
                       binning_axis=BINNING_AXIS, 
                       freq_bin=FREQ_BIN)
        pp.compute_statistics(f_binned_chain, f_analysis)