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
import postprocessing as pp
import camb
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
from scipy.stats import uniform
import pocomc
from multiprocessing import Pool

#########################################LOADING THE DATA###########################################
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run MCMC analysis with different setups.')
parser.add_argument('--data_env', type=str, required=True, help='Path to the .env file for the data')
parser.add_argument('--sampler_env', type=str, required=True, help='Path to the .env file for the sampling setup')
parser.add_argument('--machine', type =str,required=True)
parser.add_argument('--omega',default=True, type=float, required=True, help='Whether to use the reconstructed power spectrum')
args = parser.parse_args()

#Load the paths
MACHINE = args.machine
OMEGA = float(args.omega)
PATHS = data_handling.load_json_to_dict('paths.json')[MACHINE]
MAIN_DIR = PATHS['MAIN_DIR']
DATA_DIR = PATHS['DATA_DIR']
FIG_DIR = PATHS['FIG_DIR']
CHAIN_DIR = PATHS['CHAIN_DIR']

# Load the environment variables
load_dotenv(args.data_env)
load_dotenv(args.sampler_env)

#load the cosmological parameters
pars_fn = os.getenv('CAMB_INI')
pars = camb.read_ini(pars_fn)

As = pars.InitPower.As
ns = pars.InitPower.ns
k_norm = 0.05 #[1/Mpc]
h = pars.H0/100
results = camb.get_results(pars)
kh, z_pk, plin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=100, npoints = 2000)
fs8 = results.get_fsigma8()    
s8  = results.get_sigma8()     
fz = fs8/s8

if os.getenv('MODEL') == 'lin':
    def PK(k, As = As, ns = ns, amp = 0.1 , freq = OMEGA, phi = 0):
        return As*(k/0.05)**(ns-1)*(1 + np.sin(phi*np.pi+k*freq)*amp)
    prior_file = 'lin_singlepol_single_cap'
    prior_name = prior_file
    OMEGA_MIN, OMEGA_MAX = 5,150
if os.getenv('MODEL')== 'log':
    def PK(k, As = As, ns = ns, amp = 0.1 , freq = OMEGA, phi = 0):
        return As*(k/0.05)**(ns-1)*(1 + np.sin(phi*np.pi+np.log(k/0.05)*freq)*amp)
    prior_file = 'log_singlepol_single_cap'
    
# Retrieve the paths from the environment
DATA_NGC_file = os.getenv('PK_NGC_POST_DATA')
COV_NGC_file  = os.getenv('COV_NGC_POST')
DATA_NGC_file = os.path.join(DATA_DIR, DATA_NGC_file)
COV_NGC_file = os.path.join(DATA_DIR, COV_NGC_file)

# Get the prior
#Set the chains folder
OUT_FOLDER = os.getenv('OUT_FOLDER')
OUT_FOLDER = os.path.join(OUT_FOLDER, 'velo_vs_compressed',os.getenv('MODEL')+f'_{OMEGA}')

# Linear matter power spectrum (smooth and wiggly part)
PLIN = os.getenv('PLIN')
PLIN = os.path.join(MAIN_DIR, PLIN)

KMIN = float(os.getenv('KMIN')) if os.getenv('KMIN') is not None else None
KMAX = float(os.getenv('KMAX')) if os.getenv('KMAX') is not None else None
data_processor = data_handling.DataProcessor(KMIN, KMAX)
k_data,DATA_NGC = data_processor.load_data_DESI(DATA_NGC_file)
COV_NGC = data_processor.load_cov(COV_NGC_file)
invCOV = np.linalg.inv(COV_NGC)
Nmocks = 1000
Nb = len(k_data)
invCOV *= (Nmocks-Nb-2)/(Nmocks-1)

# Create the name of the data file
data_label = args.data_env.split('/')[-1].split('.')[0]

suffix = ""
common_name = f"{data_label}_{prior_name}{suffix}"

handle_log = f"{common_name}.log"
handle = f"{common_name}"

# Initialize the MCMC
mcmc = mcmc_toolkit.MCMC(1, prior_file, log_file='log/'+handle_log)

CHAIN_PATH = os.path.join(CHAIN_DIR, OUT_FOLDER, prior_name)
print(CHAIN_PATH)
# Construct the log file path inside the chains directory
log_filename = os.path.join(CHAIN_PATH, f"{handle}.log")
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
os.makedirs(CHAIN_PATH, exist_ok=True)
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
logger.info(f'DATA NGC file: {DATA_NGC_file}')
logger.info(f'COV NGC file: {COV_NGC_file}')
logger.info(f'linear matter power spectrum: {PLIN}')
logger.info(f'prior_name: {prior_name}')
logger.info(f'KMIN: {KMIN}')
logger.info(f'KMAX: {KMAX}')
logger.info(f'Filename: {common_name}')

pars.set_initial_power_function(PK)
results_feature = camb.get_results(pars)
k, _, plin_feature = results_feature.get_matter_power_spectrum(minkh=1e-4, maxkh=100, npoints = 2000)

biases = [0.71,0.26,0.67,0.52]
stoch  = [1500.,-1900.,0]
cterms = [-3.4,-1.7,6.5,0]
LPT_pars   = biases + cterms + stoch

lpt = LPT_RSD(kh,plin_feature[0],third_order=True,one_loop = True)
lpt.make_pltable(fz[0],nmax=5,apar=1,aperp=1, kv = k_data)
kl,p0_feature,p2_feature,p4_feature = lpt.combine_bias_terms_pkell(LPT_pars)


def theory(x):
    return compressed_model.Evaluate_bare(x)

def logprob(x):
    return like.logGaussian(x)
    
compressed_model = ps_constructor.PowerSpectrumConstructor(k_data, ps_filename=os.getenv('PLIN'), pf_model= os.getenv('MODEL'), ps_style='compressed')
like = likelihood.likelihoods(theory, p0_feature, invCOV)

mcmc.prior_bounds[0][mcmc.id_map['omega']] = 1
mcmc.prior_bounds[1][mcmc.id_map['omega']] = 50
loc = mcmc.prior_bounds[0]
scale = mcmc.prior_bounds[1] - mcmc.prior_bounds[0]
prior = pocomc.Prior([uniform(loc[i], scale[i]) for i in range(len(loc))])

sampler = pocomc.Sampler(
    prior=prior,
    likelihood=logprob,
    vectorize=False,
    random_state=0,
    n_effective = 800,
    n_active = None,
    output_dir = CHAIN_PATH,
    output_label = 'state_saved'
)

sampler.run(save_every=10)
samples, weights, logl, logp = sampler.posterior()
np.save(os.path.join(CHAIN_PATH,'sample.npy'), samples)
np.save(os.path.join(CHAIN_PATH,'weights.npy'), weights)
np.save(os.path.join(CHAIN_PATH,'logl.npy'), logl)
np.save(os.path.join(CHAIN_PATH,'logp.npy'), logp)
np.save(os.path.join(CHAIN_PATH,'params.npy'), LPT_pars)

import corner
import matplotlib.pyplot as plt

fig = corner.corner(samples, weights=weights, color="C0")
plt.savefig(os.path.join(CHAIN_PATH,'corner.png'))
