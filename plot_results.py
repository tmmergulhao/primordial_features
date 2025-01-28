import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def plot_main(ax, k, data, bf, bf_smooth, bf_bao, ylabel=None):
    """Plot main data with smoothed and BAO models."""
    ax.plot(k, data / bf_smooth, marker='o', label="Data")
    ax.plot(k, bf / bf_smooth, color='blue', lw=2, label="PF")
    ax.plot(k, bf_bao / bf_smooth, color='red', lw=2, label="BAO")
    ax.set_xlim((k[0], k[-1]))
    ax.set_ylim((0.8, 1.2))
    ax.grid()
    ax.legend()
    ax.set_xlabel(r'k [h/Mpc]')
    if ylabel:
        ax.set_ylabel(ylabel)

def plot_main_BAO(ax, k, data,bf_smooth, bf_bao, ylabel=None):
    """Plot main data with smoothed and BAO models."""
    ax.plot(k, data / bf_smooth, marker='o', label="Data")
    ax.plot(k, bf_bao / bf_smooth, color='red', lw=2, label="BAO")
    ax.set_xlim((k[0], k[-1]))
    ax.set_ylim((0.8, 1.2))
    ax.grid()
    ax.legend()
    ax.set_xlabel(r'k [h/Mpc]')
    if ylabel:
        ax.set_ylabel(ylabel)

def plot_residuals(ax, k, data, bf, cov, bf_bao):
    """Plot residuals."""
    residuals = (bf - data) / np.sqrt(np.diag(cov))
    residuals_bao = (bf_bao - data) / np.sqrt(np.diag(cov))
    ax.scatter(k, residuals, color='blue', s=1, marker='s', label="Residuals")
    ax.scatter(k, residuals_bao, color='red', s=1, marker='v', label="Residuals")
    ax.axhline(0, color='black', lw=1, linestyle='--')
    ax.set_xlim((k[0], k[-1]))
    ax.set_ylim((-5, 5))
    ax.set_ylabel("Residuals")
    ax.grid()

def plot_residuals_BAO(ax, k, data, cov, bf_bao):
    """Plot residuals."""
    residuals_bao = (bf_bao - data) / np.sqrt(np.diag(cov))
    ax.scatter(k, residuals_bao, color='red', s=1, marker='v', label="Residuals")
    ax.axhline(0, color='black', lw=1, linestyle='--')
    ax.set_xlim((k[0], k[-1]))
    ax.set_ylim((-5, 5))
    ax.set_ylabel("Residuals")
    ax.grid()

def plot_results(mcmc, likelihood, theory, DATA, COV_NGC, COV_SGC, k, FIG_PATH, primordialfeature_model,
                 handle,save_chi2=False):
    """
    Plot the results of MCMC chains after convergence.
    
    Parameters:
        mcmc: Object with MCMC results and methods (e.g., `get_ML`).
        theory: Function to compute the theoretical predictions.
        DATA: Full dataset split into NGC and SGC components.
        COV_NGC, COV_SGC: Covariance matrices for NGC and SGC.
        k: Array of wavenumbers.
        FIG_PATH: Path to save the output plot.
        handle: Identifier for the current MCMC run.
    """
    # Load best fit and related parameters
    theta_ML = mcmc.get_ML(handle=handle, gelman_rubin=mcmc.gelman_rubin)
    chi2 = likelihood.chi2(theta_ML)
    
    if save_chi2:
        with open(os.path.join(FIG_PATH,'chi2_'+handle + '.txt'), 'w') as f:
            f.write(f"chi2: {chi2:.2f}")

    # Modify parameters for smooth and BAO cases
    theta_smooth_ML = theta_ML.copy()
    theta_smooth_ML[mcmc.id_map['sigma_nl']] = 1000
    if primordialfeature_model != 'None':
        theta_BAO = theta_ML.copy()
        theta_BAO[mcmc.id_map['A']] = 0
        
        # Compute theoretical predictions
        bf_NGC, bf_SGC = theory(theta_ML)[:len(k)], theory(theta_ML)[len(k):]
        bf_smooth_NGC, bf_smooth_SGC = theory(theta_smooth_ML)[:len(k)], theory(theta_smooth_ML)[len(k):]
        bf_BAO_NGC, bf_BAO_SGC = theory(theta_BAO)[:len(k)], theory(theta_BAO)[len(k):]
        
        # Split dataset
        DATA_NGC, DATA_SGC = DATA[:len(k)], DATA[len(k):]

        # Create figure and grid layout
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0)

        # Main plot and residuals for NGC
        ax_main_ngc = fig.add_subplot(gs[0, 0])
        ax_main_ngc.set_title(f"emcee chi2/dof: {chi2:.2f} / {2 * len(k) - mcmc.ndim:.2f}")
        plot_main(ax_main_ngc, k, DATA_NGC, bf_NGC, bf_smooth_NGC, bf_BAO_NGC, ylabel=r'P(k) / P(k) smooth')

        ax_res_ngc = fig.add_subplot(gs[1, 0], sharex=ax_main_ngc)
        plot_residuals(ax_res_ngc, k, DATA_NGC, bf_NGC, COV_NGC, bf_BAO_NGC)

        # Main plot and residuals for SGC
        ax_main_sgc = fig.add_subplot(gs[0, 1])
        ax_main_sgc.set_title(f'A = {theta_ML[mcmc.id_map["A"]]:.2f}, omega= {theta_ML[mcmc.id_map["omega"]]:.2f}')
        plot_main(ax_main_sgc, k, DATA_SGC, bf_SGC, bf_smooth_SGC, bf_BAO_SGC, ylabel=r'P(k) / P(k) smooth')

        ax_res_sgc = fig.add_subplot(gs[1, 1], sharex=ax_main_sgc)
        plot_residuals(ax_res_sgc, k, DATA_SGC, bf_SGC, COV_SGC, bf_BAO_SGC)
        
        # Save the figure
        plt.savefig(os.path.join(FIG_PATH, handle + '.png'))
        plt.close(fig)
    else:
        theta_BAO = theta_ML

        # Compute theoretical predictions
        bf_smooth_NGC, bf_smooth_SGC = theory(theta_smooth_ML)[:len(k)], theory(theta_smooth_ML)[len(k):]
        bf_BAO_NGC, bf_BAO_SGC = theory(theta_BAO)[:len(k)], theory(theta_BAO)[len(k):]
        
        # Split dataset
        DATA_NGC, DATA_SGC = DATA[:len(k)], DATA[len(k):]

        # Create figure and grid layout
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0)

        # Main plot and residuals for NGC
        ax_main_ngc = fig.add_subplot(gs[0, 0])
        ax_main_ngc.set_title(f"emcee chi2/dof: {chi2:.2f} / {2 * len(k) - mcmc.ndim:.2f}")
        plot_main_BAO(ax_main_ngc, k, DATA_NGC,bf_smooth_NGC, bf_BAO_NGC, ylabel=r'P(k) / P(k) smooth')

        ax_res_ngc = fig.add_subplot(gs[1, 0], sharex=ax_main_ngc)
        plot_residuals_BAO(ax_res_ngc, k, DATA_NGC,COV_NGC, bf_BAO_NGC)

        # Main plot and residuals for SGC
        ax_main_sgc = fig.add_subplot(gs[0, 1])
        plot_main_BAO(ax_main_sgc, k, DATA_SGC, bf_smooth_SGC, bf_BAO_SGC, ylabel=r'P(k) / P(k) smooth')

        ax_res_sgc = fig.add_subplot(gs[1, 1], sharex=ax_main_sgc)
        plot_residuals_BAO(ax_res_sgc, k, DATA_SGC, COV_SGC, bf_BAO_SGC)
        
        # Save the figure
        plt.savefig(os.path.join(FIG_PATH, handle + '.png'))
        plt.close(fig)