import numpy as np
import os, sys
sys.path.append("/home/tmergulhao/primordial_features")
import postprocessing as pp

# Directories and filenames
chains_dir = '/home/tmergulhao/chains/'
analysis_dir = '/home/tmergulhao/PF_DESI_RESULTS/'
s_template = 'DESI_Y1_DATA_ELG2/lin_singlepol/{}.0_{}.0/DATA_DESI_ELG2_DATA_prerecon_lin_singlepol_{}.0_{}.0_'
freqs = [[100,900],[800,2000],[1900,3000],[2900,4000]]

for freq in freqs:
    for i in range(0,1):
        # Number of chains to load
        n = 4
        burn_in = 0.3
        freq_ranges = [freq]
        s = s_template.format(freq[0], freq[1], freq[0], freq[1])

        # File paths
        f_bare = chains_dir + s.format(i)
        f_total_chain = analysis_dir + s.format(i) + '.h5'
        f_binned_chain = analysis_dir + s.format(i) + 'binned.h5'
        f_analysis = analysis_dir + s.format(i) + 'analysis.h5'
        
        # Ensure directories exist for the output files
        os.makedirs(os.path.dirname(f_total_chain), exist_ok=True)
        
        # Process chains
        pp.get_total_chain(f_bare, f_total_chain, n, burnin_frac=burn_in, thin=10)
        pp.BinnedChain([f_total_chain], freq_ranges, f_binned_chain, binning_id=11, freq_bin=10)
        pp.compute_statistics(f_binned_chain, f_analysis)