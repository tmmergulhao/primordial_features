import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from iminuit import Minuit

# Parse command-line arguments **before** importing `main.py`
parser = argparse.ArgumentParser(description="Run Minuit fits over frequency bins.")

parser.add_argument('--data_env', required=True, help="Path to the data environment file.")
parser.add_argument('--sampler_env', required=True, help="Path to the sampler environment file.")
parser.add_argument('--mock', required=True, type=str, help="Mock number.")
parser.add_argument('--machine', required=True, help="Machine type (e.g., MAC).")
parser.add_argument('--reconstruction', required=True, type=str, help="Whether to use reconstruction (True/False).")
parser.add_argument('--omega_min', required=True, type=float, help="Minimum omega value.")
parser.add_argument('--omega_max', required=True, type=float, help="Maximum omega value.")
args = parser.parse_args()

import sys
sys.argv = [
    'main.py',
    '--data_env', args.data_env,
    '--sampler_env', args.sampler_env,
    '--mock', args.mock,
    '--machine', args.machine,
    '--reconstruction', args.reconstruction,
    '--omega_min', str(args.omega_min),
    '--omega_max', str(args.omega_max),
]

from main import *

# Read the frequency bin size from the environment variable
freq_bin = int(os.getenv('FREQ_BIN'))  # Default to 10 if not set

# Extract values from parsed arguments
omega_min = args.omega_min
omega_max = args.omega_max

# Define the chi2 function
def chi2(*theta):
    return PrimordialFeature_likelihood.chi2(list(theta))

# Define frequency bins
bins = np.arange(omega_min, omega_max, freq_bin)

# Prepare to collect results for each frequency bin
results = []

# Get the parameter limits
pf_limits = [(a, b) for a, b in zip(mcmc.prior_bounds[0], mcmc.prior_bounds[1])]

def run_fit(f):
    omega_bin_center = f + freq_bin / 2.0
    logger.info(f"Running Minuit fit for frequency bin {omega_bin_center:.2f}...")
    # Initial guess for minimization
    PF_guess = np.array([2,2,0,0,0,0,0,1,3,3,0,omega_bin_center,0.25])

    # Create Minuit object with the updated guess and limits
    m_PF = Minuit(chi2, name=mcmc.labels,
                  **{label: val for label, val in zip(mcmc.labels, PF_guess)})

    # Update parameter limits for omega
    local_limits = pf_limits.copy()
    local_limits[mcmc.id_map['omega']] = (omega_bin_center, omega_bin_center)
    m_PF.limits = local_limits

    # Run MIGRAD minimization
    m_PF.migrad(ncall=20000)

    # Extract best-fit parameters and chi2 values
    theta_PF_ML = list(m_PF.values.to_dict().values())
    chi2_PF = m_PF.fval

    return {
        "omega_bin_center": omega_bin_center,
        "theta_PF": theta_PF_ML,
        "chi2_PF": chi2_PF
    }

# Run in parallel using multiprocessing.Pool with 5 processes
if __name__ == "__main__":
    out_file = CHAIN_PATH+"/minuit_fit_results.json"

    with Pool(processes=5) as pool:
        results = list(pool.imap(run_fit, bins))

    # Save the results to the specified output file
    with open(out_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Results saved to {out_file}")