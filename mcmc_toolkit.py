import os, sys, json, emcee
import numpy as np
import logging
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from getdist import plots, MCSamples

@dataclass
class MCMC:
    nwalkers: int
    prior_name: str
    priors_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), 'priors'))
    burnin_frac: float = 0.5
    log_file: Optional[str] = None
    prior_dictionary: Dict[str, Any] = field(init=False)
    ndim: int = field(init=False)
    prior_bounds: np.ndarray = field(init=False)
    labels: List[str] = field(init=False)

    def __post_init__(self):
        # Configure logging
        if self.log_file:
            logging.basicConfig(filename=self.log_file, level=logging.INFO,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        directory = self.priors_dir
        try:
            with open(os.path.join(directory, f'{self.prior_name}.json')) as json_file:
                dic = json.load(json_file, object_pairs_hook=OrderedDict)
        except FileNotFoundError:
            self.logger.error(f'File {self.prior_name}.json not found in {directory}.')
            sys.exit(-1)
        except json.JSONDecodeError:
            self.logger.error(f'Problem decoding the JSON file {self.prior_name}.json.')
            sys.exit(-1)
        except Exception as e:
            self.logger.error(f'An unexpected error occurred: {e}')
            sys.exit(-1)

        self.prior_dictionary = dic  # Save the prior dictionary
        self.logger.info(f'Using {self.prior_name} file')
        self.ndim = len(dic)
        self.prior_bounds = np.zeros((2, self.ndim))
        self.labels = []

        for index, key in enumerate(dic.keys()):
            self.labels.append(key)
            self.prior_bounds[0, index], self.prior_bounds[1, index] = dic[key]

    def set_walkers(self, nwalkers: int) -> None:
        """
        Allow the user to change the number of walkers after the class is initialized.

        Args:
            nwalkers (int): Number of walkers to be used in the MCMC analysis.
        """
        if nwalkers <= 0:
            raise ValueError("Number of walkers must be a positive integer.")
        self.nwalkers = nwalkers
        self.logger.info(f'Number of walkers set to {self.nwalkers}')

    def set_burnin_frac(self, burnin_frac: float) -> None:
        """
        Allow the user to change the usual burnin_frac used in the analysis. The standard value is 
        0.5 (i.e., 50% of the chain is discarded).

        Args:
            burnin_frac (float): The percentage of the chain that must be discarded when 
            performing some analysis.
        """
        if not (0.0 <= burnin_frac <= 1.0):
            raise ValueError("burnin_frac must be between 0.0 and 1.0.")
        self.burnin_frac = burnin_frac
        self.logger.info(f'Burn-in fraction set to {self.burnin_frac}')

    def create_walkers(self, mode: str, file: bool = False, x0: np.ndarray = None, sigmas: 
        np.ndarray = None, ranges: np.ndarray = None) -> np.ndarray:
        """
        Create the walkers following three different recipes. Each mode will require a different 
        set of inputs.

        Args:
            mode (str): The name of the recipe to be used. Options:
                1) 'gaussian': Distribute the walkers following a Gaussian distribution with mean 
                x0 (array) and variance sigma (array).
                You need to provide x0 and sigmas.
                2) 'uniform_prior': Distribute the walkers following a uniform distribution inside 
                the parameter boundaries defined in the prior file.
                No additional input is needed.
                3) 'uniform_thin': Distribute the walkers uniformly within a specified range around 
                x0. You need to provide x0 and ranges.

            file (bool or str, optional): Whether to save the initial positions in a .txt file. If 
            a string is provided, it will be used as the filename prefix. Defaults to False. 
            
            x0 (np.ndarray, optional): Used in the 'gaussian' and 'uniform_thin' recipes. Defaults 
            to None.

            sigmas (np.ndarray, optional): Used in the 'gaussian' recipe. Defaults to None.

            ranges (np.ndarray, optional): Used in the 'uniform_thin' recipe. Defaults to None.

        Returns:
            np.ndarray: A 2D array with the initial positions of the walkers.
        """
        pos = np.zeros((self.nwalkers, self.ndim))

        if mode == 'uniform_prior':
            self.logger.info('Using uniform prior')
            for i in range(self.ndim):
                pos[:, i] = np.random.uniform(self.prior_bounds[0, i], self.prior_bounds[1, i], 
                self.nwalkers)

                self.logger.info(f'For param {self.labels[i]}: Minimum: {round(np.min(pos[:, i]),2)} | Maximum: {round(np.max(pos[:, i]), 2)}')

        elif mode == 'gaussian':
            if x0 is None or sigmas is None:
                raise ValueError("x0 and sigmas must be provided for 'gaussian' mode.")
            for i in range(self.ndim):
                pos[:, i] = sigmas[i] * np.random.randn(self.nwalkers) + x0[i]
                self.logger.info(f'For param {self.labels[i]}: Minimum: {round(np.min(pos[:, i]), 2)} | Maximum: {round(np.max(pos[:, i]), 2)}')

        elif mode == 'uniform_thin':
            if x0 is None or ranges is None:
                raise ValueError("x0 and ranges must be provided for 'uniform_thin' mode.")
            self.logger.info('Using the uniform_thin walker positioning')
            lower = x0 - ranges
            upper = x0 + ranges
            for i in range(self.ndim):
                pos[:, i] = np.random.uniform(lower[i], upper[i], self.nwalkers)
                self.logger.info(f'For param {self.labels[i]}: Minimum: {round(np.min(pos[:, i]), 2)} | Maximum: {round(np.max(pos[:, i]), 2)}')

        else:
            raise ValueError(f"Unknown mode '{mode}'. Valid options are 'gaussian', 'uniform_prior', and 'uniform_thin'.")

        if isinstance(file, str):
            try_mkdir('initial_positions')
            filename = os.path.join(os.getcwd(), 'initial_positions', f'{file}_initial_pos.txt')
            np.savetxt(filename, pos)
            self.logger.info(f'Initial positions saved to {filename}')

        return pos

    def gelman_rubin_convergence(self, within_chain_var: np.ndarray, mean_chain: np.ndarray, 
        chain_length: int) -> np.ndarray:
        """
        Calculate the Gelman & Rubin diagnostic.

        Args:
            within_chain_var (np.ndarray): Within-chain variances.
            mean_chain (np.ndarray): Mean of the chains.
            chain_length (int): Length of the chains.

        Returns:
            np.ndarray: Potential scale reduction parameter (R-hat).
        """
        Nchains = within_chain_var.shape[0]
        dim = within_chain_var.shape[1]
        meanall = np.mean(mean_chain, axis=0)
        W = np.mean(within_chain_var, axis=0)
        B = np.zeros(dim, dtype=np.float64)
        
        for jj in range(Nchains):
            B += chain_length * (meanall - mean_chain[jj])**2 / (Nchains - 1)
        
        estvar = (1 - 1 / chain_length) * W + B / chain_length
        R_hat = np.sqrt(estvar / W)
        
        self.logger.info(f'Gelman-Rubin diagnostic calculated: {R_hat}')
        
        return R_hat

    def prep_gelman_rubin(self, sampler):
        """
        Prepare data for Gelman & Rubin diagnostic calculation.

        Args:
            sampler: MCMC sampler object with a `get_chain` method.

        Returns:
            tuple: within_chain_var (np.ndarray), mean_chain (np.ndarray), chain_length (int)
        """
        chain = sampler.get_chain()
        chain_length = chain.shape[0]
        chainsamples = chain[int(chain_length / 2):, :, :].reshape((-1, self.ndim))
        within_chain_var = np.var(chainsamples, axis=0)
        mean_chain = np.mean(chainsamples, axis=0)
        
        self.logger.info('Prepared data for Gelman-Rubin diagnostic calculation')
        
        return within_chain_var, mean_chain, chain_length

    def plot_walkers(self,samplers,name, save = True):
        '''
        Get a list of samplers and make the 1D plot. The input must be an array [n_samplers]
        '''
        #Get the number of samplers
        N_samples = len(samplers)

        #Start the figure
        fig, axes = plt.subplots(self.ndim, figsize=(16, self.ndim*3), sharex=True)

        #Get the colors for the plots
        color=cm.brg(np.linspace(0,1,N_samples))

        #Iterate over the backends and plot the walkers
        for i in range(0,N_samples):
            chain = samplers[i].get_chain()
            for index,this_param in enumerate(self.labels):
                ax = axes[index]
                ax.plot(chain[:,:,index],alpha=0.5,color=color[i])
                ax.set_ylabel(this_param,size=35)
            del chain

        if save:
            fig.tight_layout()
            #directory to figures
            this_dir = os.getcwd()
            figures_dir = this_dir+'/figures/'
            try_mkdir('figures')
            plt.savefig(figures_dir+name+'_walkers.pdf')
            plt.close('all')
        else:
            plt.show()

    def plot_1d(self, samplers: List, name: str) -> None:
        """
        Plot 1D distributions for the given samplers.

        Args:
            samplers (List): List of sampler objects.
            name (str): Name for the saved plot file.
        """
        plot_settings = {
            'ignore_rows': 0.5,
            'fine_bins': 1000,
            'fine_bins_2D': 2000,
            'smooth_scale_1D': 0.3,
        }
        N_samples = len(samplers)
        samples = []
        chain = samplers[0].get_chain()
        chain_length = chain.shape[0]
        del chain

        for i in range(N_samples):
            chain = samplers[i].get_chain(flat=True)
            samples.append(MCSamples(samples=chain, labels=self.labels, names=self.labels, 
            settings=plot_settings))
        del chain

        g1 = plots.get_subplot_plotter(width_inch=20)
        g1.settings.legend_fontsize = 20
        g1.settings.axes_fontsize = 20
        g1.settings.axes_labelsize = 20
        g1.settings.title_limit = True
        g1.plots_1d(samples)

        figures_dir = os.path.join(os.getcwd(), 'figures')
        try_mkdir('figures')
        g1.export(os.path.join(figures_dir, f'{name}_1D_ALL.png'))
        plt.close('all')
        self.logger.info(f'1D plot saved to {figures_dir}/{name}_1D_ALL.png')

    def plot_corner(self, handle: str, gelman: Optional[Dict] = None, save: Optional[str] = None, 
    width_inch: int = 15, ranges: Dict = {}, plot_settings: Dict = {'fine_bins': 1000, 
    'fine_bins_2D': 1500, 'smooth_scale_1D': 0.3, 'smooth_scale_2D': 0.2}) -> None:
        """
        Plot a corner plot for the given chains.

        Args:
            handle (str): Handle for the chain files.
            gelman (Dict, optional): Gelman-Rubin diagnostic results. Defaults to None.
            save (str, optional): Name for the saved plot file. Defaults to None.
            width_inch (int, optional): Width of the plot in inches. Defaults to 15.
            ranges (Dict, optional): Ranges for the plot. Defaults to {}.
            plot_settings (Dict, optional): Settings for the plot. Defaults to 
            {'fine_bins': 1000, 
            'fine_bins_2D': 1500, 
            'smooth_scale_1D': 0.3, 
            'smooth_scale_2D': 0.2}.
        """
        chain_dir = os.path.join(os.getcwd(), 'chains')
        
        if gelman is not None:
            N_chains = gelman['N']
            for i in range(N_chains):
                name = os.path.join(chain_dir, f'{handle}Run_{i}.h5')
                backend = emcee.backends.HDFBackend(name, read_only=True)
                chain = backend.get_chain(flat=False)
                chain_size = chain.shape[0]
                burnin = int(self.burnin_frac * chain_size)
                chain = backend.get_chain(flat=True, discard=burnin)

                if i == 0:
                    final_chain = chain
                else:
                    final_chain = np.vstack((final_chain, chain))
        else:
            name = os.path.join(chain_dir, f'{handle}.h5')
            backend = emcee.backends.HDFBackend(name, read_only=True)
            chain = backend.get_chain(flat=False)
            chain_size = chain.shape[0]
            burnin = int(self.burnin_frac * chain_size)
            final_chain = backend.get_chain(flat=True, discard=burnin)

        samples = MCSamples(samples=final_chain, labels=self.labels, names=self.labels, 
        settings=plot_settings, ranges=ranges)

        g1 = plots.get_subplot_plotter(width_inch=width_inch)
        g1.settings.legend_fontsize = 20
        g1.settings.axes_fontsize = 20
        g1.settings.axes_labelsize = 20
        g1.settings.title_limit = True
        g1.settings.progress = True
        g1.triangle_plot(samples)

        if save is not None:
            figures_dir = os.path.join(os.getcwd(), 'figures')
            try_mkdir('figures')
            plt.savefig(os.path.join(figures_dir, f'{save}.png'))
            plt.close('all')
            self.logger.info(f'Corner plot saved to {figures_dir}/{save}.png')

    def plot_CorrMatrix(self, handle: str, gelman: Optional[Dict] = None) -> None:
        """
        Plot the correlation matrix for the given chains.

        Args:
            handle (str): Handle for the chain files.
            gelman (Dict, optional): Gelman-Rubin diagnostic results. Defaults to None.
        """
        chain_dir = os.path.join(os.getcwd(), 'chains')

        if gelman is not None:
            N_chains = gelman['N']
            for i in range(N_chains):
                name = os.path.join(chain_dir, f'{handle}Run_{i}.h5')
                backend = emcee.backends.HDFBackend(name, read_only=True)
                chain = backend.get_chain(flat=False)
                chain_size = chain.shape[0]
                burnin = int(self.burnin_frac * chain_size)
                chain = backend.get_chain(flat=True, discard=burnin)

                if i == 0:
                    final_chain = chain
                else:
                    final_chain = np.vstack((final_chain, chain))
        else:
            name = os.path.join(chain_dir, f'{handle}.h5')
            backend = emcee.backends.HDFBackend(name, read_only=True)
            chain = backend.get_chain(flat=False)
            chain_size = chain.shape[0]
            burnin = int(self.burnin_frac * chain_size)
            final_chain = backend.get_chain(flat=True, discard=burnin)

        fig, ax1 = plt.subplots(1, 1, figsize=(9, 9))
        im = ax1.imshow(np.corrcoef(final_chain.T), cmap=plt.get_cmap('RdBu'))

        ax1.set_xticks(np.arange(-0.5, self.ndim - 1 + 1, 1), minor=False)
        ax1.set_yticks(np.arange(-0.5, self.ndim - 1 + 1, 1), minor=False)
        ax1.set_xticklabels([], minor=False)
        ax1.set_yticklabels([], minor=False)

        ax1.set_xticks(np.arange(0, self.ndim, 1), minor=True)
        ax1.set_xticklabels(['$' + x + '$' for x in self.labels], minor=True)
        ax1.set_yticks(np.arange(0, self.ndim, 1), minor=True)
        ax1.set_yticklabels(['$' + x + '$' for x in self.labels], minor=True)
        ax1.grid(linewidth=10, color='white')
        fig.colorbar(im)

        figures_dir = os.path.join(os.getcwd(), 'figures')
        try_mkdir('figures')
        plt.savefig(os.path.join(figures_dir, f'{handle}_Corr.png'))
        plt.close('all')
        self.logger.info(f'Correlation matrix plot saved to {figures_dir}/{handle}_Corr.png')

    def in_prior(self, x: np.ndarray, params: Optional[List[str]] = None) -> bool:
        """
        Return True if parameters are inside priors and False otherwise.

        Args:
            x (np.ndarray): Array of parameter values to check.
            params (List[str], optional): List of parameter labels to check. If None, all parameters
             are checked. Defaults to None.

        Returns:
            bool: True if all specified parameters are within their prior bounds, False otherwise.
        """
        if params is None:
            for i, this_param in enumerate(self.prior_bounds.T):
                this_value = x[i]
                this_lower_bound, this_upper_bound = this_param
                if not (this_lower_bound < this_value < this_upper_bound):
                    return False
            return True
        else:
            for this_param in params:
                this_index = self.labels.index(this_param)
                this_lower_bound, this_upper_bound = self.prior_dictionary[this_param]
                if not (this_lower_bound < x[this_index] < this_upper_bound):
                    return False
            return True


    def run(self, name: str, steps: int, pos: np.ndarray, loglikelihood: Callable, 
    pool: Optional[Any] = None, new: bool = True, plots: bool = False, 
    args: Optional[List[Any]] = None, a: float = 2, metric_interval: int = 25) -> None:
        """
        Run the MCMC simulation with optional MPI support.

        Args:
            name (str): Name for the chain files.
            steps (int): Number of steps to run the MCMC.
            pos (np.ndarray): Initial positions of the walkers.
            loglikelihood (Callable): Log-likelihood function.
            pool (Optional[Any], optional): Pool for parallel processing. Defaults to None.
            new (bool, optional): Whether to start a new run or continue from the last sample.
            Defaults to True.
            plots (bool, optional): Whether to generate plots. Defaults to False.
            args (List[Any], optional): Additional arguments for the log-likelihood function. 
            Defaults to None.
            a (float, optional): Stretch move parameter. Defaults to 2.
            metric_interval (int, optional): Interval at which to calculate and save metrics. 
            Defaults to 25.
        """
        try_mkdir('chains')
        chain_dir = os.path.join(os.getcwd(), 'chains')
        filename = os.path.join(chain_dir, f'{name}.h5')
        backend = emcee.backends.HDFBackend(filename)

        autocorr = []
        acceptance = []

        # Load previous autocorr and acceptance data if available
        try:
            autocorr = np.loadtxt(os.path.join(chain_dir, f'{name}_tau.txt')).tolist()
            acceptance = np.loadtxt(os.path.join(chain_dir, f'{name}_acceptance.txt')).tolist()
        except OSError:
            pass

        sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, loglikelihood, args=args, backend=backend,
            moves=[emcee.moves.StretchMove(a=a)], pool=pool
        )

        if new:
            initial_sample = pos
        else:
            initial_sample = sampler.get_last_sample()

        for sample in sampler.sample(initial_sample, iterations=steps, progress=True):
            if sampler.iteration % metric_interval == 0:
                try:
                    tau = sampler.get_autocorr_time(tol=0)
                    autocorr.append(np.mean(tau))
                    acceptance.append(np.mean(sampler.acceptance_fraction))
                    np.savetxt(os.path.join(chain_dir, f'{name}_tau.txt'), autocorr)
                    np.savetxt(os.path.join(chain_dir, f'{name}_acceptance.txt'), acceptance)
                    self.logger.info(f'Mean acceptance fraction: {np.mean(sampler.acceptance_fraction)}')
                    self.logger.info(f'Mean autocorrelation time: {np.mean(tau)}')
                except emcee.autocorr.AutocorrError:
                    self.logger.warning('Autocorrelation time could not be estimated.')

        if plots:
            self.plot_walkers([sampler], name)
            self.plot_1d([sampler], name)

    def run_gelman(self,gelman_rubins,handle,loglikelihood,x0=None,sigmas=None,new_run = True,
    args=None,ranges=None, a=2, save_epsilon = False):
        '''
        This function start the chains and only stop when convergence criteria is achieved
        '''

        #Directory where the chains will be storaged
        try_mkdir('chains')
        directory_to_chain = os.getcwd()+"/chains/"

        #Read the Convergence Parameters from a dictionary
        try:
            N = gelman_rubins['N']
            epsilon = gelman_rubins['epsilon']
            minlength = gelman_rubins['min_length']
            convergence_steps = gelman_rubins['convergence_steps']
            initial_option = gelman_rubins['initial']
        except:
            print('Problem reading the Gelman-Rubin convergence parameters!')
            print('keys: N, epsilon, min_length, convergence_steps, initial')
            sys.exit(-1)

        #List containing all the samplers
        list_samplers = []

        #storate values used to estimate convergence
        within_chain_var = np.zeros((N, self.ndim))
        mean_chain = np.zeros((N, self.ndim))
        chain_length = 0
        scalereduction = np.arange(self.ndim, dtype=np.float)
        scalereduction.fill(2.)

        #Counting the number of iterations:
        counter = 0

        print('You are considering', minlength, 'as the minimum lenght for the chain')
        print('Convergence test happens every', convergence_steps, 'steps')
        print('Number of walkers:', self.nwalkers)
        print('Number of Parameters:', self.ndim)
        print('Number of parallel chains:', N)

        #ask_to_continue()

        #Create all the samplers and their walkers
        for i in range(0,N):
            #create the backend
            filename = directory_to_chain+handle+'Run_'+str(i)+'.h5'
            backend   = emcee.backends.HDFBackend(filename)
            if args is not None:
                list_samplers.append(emcee.EnsembleSampler(self.nwalkers,self.ndim,loglikelihood,
                args=args,backend=backend,moves=[emcee.moves.StretchMove(a=a)]))
            else:
                list_samplers.append(emcee.EnsembleSampler(self.nwalkers,self.ndim,loglikelihood,
                backend=backend,moves=[emcee.moves.StretchMove(a=a)]))

        #Kicking off all chains to have the minimum length
        if new_run:
            for i in range(0,N):
                to_print = 'Preparing chain '+str(i)
                print(to_print.center(80, '*'))
                name_initial_pos = handle+'_initial_pos_Run_'+str(i)
                print('Positions for the chain', i)
                pos = self.create_walkers(initial_option,file=name_initial_pos,x0=x0,sigmas=sigmas,
                ranges=ranges)
                print('Go!')
                list_samplers[i].run_mcmc(pos,minlength,progress=True)
                within_chain_var[i],mean_chain[i],chain_length = self.prep_gelman_rubin(list_samplers[i])

        else:
            for i in range(0,N):
                to_print = 'Preparing chain '+str(i)
                print(to_print.center(80, '*'))
                print('Go!')
                list_samplers[i].run_mcmc(None,minlength,progress=True)
                within_chain_var[i],mean_chain[i],chain_length = \
                    self.prep_gelman_rubin(list_samplers[i])
        '''
        At this points all chains have the same length. It is checked if they already converged. 
        If that is not the case they continue to run
        '''
        print('All chains with the minimum length!')
        print('Checking convergence...')
        plotname = handle+'_'+str(counter)
        self.plot_1d(list_samplers,plotname)
        scalereduction = self.gelman_rubin_convergence(within_chain_var,mean_chain,chain_length/2)
        eps = abs(1-scalereduction)

        print('epsilon = ', eps)

        if any(eps > epsilon):
            print('Did not converge! Running more steps...')

        '''
        If the minimum length was not enough, more steps are done. As soon as the epsilon achieves 
        crosses the threshold, the analysis is done.
        '''
        while any(eps > epsilon):
            counter += 1
            print('Running iteration', counter)
            for i in range(0,N):
                list_samplers[i].run_mcmc(None,convergence_steps,progress=True)
                within_chain_var[i],mean_chain[i],chain_length = self.prep_gelman_rubin(list_samplers[i])
            scalereduction = self.gelman_rubin_convergence(within_chain_var,mean_chain,chain_length/2)
            eps = abs(1-scalereduction)

            print('epsilon = ',eps)

            plotname = handle+'_'+str(counter)
            self.plot_1d(list_samplers,plotname)

        print('Convergence Achieved!')
        print('Plotting walkers position over steps...')
        self.plot_walkers(list_samplers,plotname)
        print('Plotting the correlation matrix...')
        self.plot_CorrMatrix(handle = handle, gelman=gelman_rubins)
        print('Making a corner plot...')
        self.plot_corner(handle = handle, gelman = gelman_rubins, save = handle+"_Corner")
        print('Done!')
        
    def get_chain(self, handle: str, gelman: Optional[Dict] = None) -> np.ndarray:
            """
            Retrieve the MCMC chain.

            Args:
                handle (str): Handle used in the MCMC analysis.
                gelman (Dict, optional): Dictionary used as input for the Gelman-Rubin convergence 
                criteria. Defaults to None.

            Returns:
                np.ndarray: An array with the parameter samples.
            """
            chain_dir = os.path.join(os.getcwd(), 'chains')

            if gelman is not None:
                N_chains = gelman['N']
                for i in range(N_chains):
                    name = os.path.join(chain_dir, f'{handle}Run_{i}.h5')
                    backend = emcee.backends.HDFBackend(name, read_only=True)
                    chain = backend.get_chain(flat=False)
                    chain_size = chain.shape[0]
                    burnin = int(self.burnin_frac * chain_size)
                    chain = backend.get_chain(flat=True, discard=burnin)
                    if i == 0:
                        final_chain = chain
                    else:
                        final_chain = np.vstack((final_chain, chain))
            else:
                name = os.path.join(chain_dir, f'{handle}.h5')
                backend = emcee.backends.HDFBackend(name, read_only=True)
                chain = backend.get_chain(flat=False)
                chain_size = chain.shape[0]
                burnin = int(self.burnin_frac * chain_size)
                final_chain = backend.get_chain(flat=True, discard=burnin)

            return final_chain


    def get_ML(self, handle: str, gelman: Optional[Dict] = None) -> np.ndarray:
        """
        Search in the total sample of walker positions for the set of parameters that gives the 
        best fit to the data.

        Args:
            handle (str): Handle used in the MCMC analysis.
            gelman (Dict, optional): Dictionary used as input for the Gelman-Rubin convergence 
            criteria. Defaults to None.

        Returns:
            np.ndarray: An array with the parameters that give the best fit to the data.
        """
        chain_dir = os.path.join(os.getcwd(), 'chains')

        if gelman is not None:
            N_chains = gelman['N']
            for i in range(N_chains):
                name = os.path.join(chain_dir, f'{handle}Run_{i}.h5')
                backend = emcee.backends.HDFBackend(name, read_only=True)
                chain = backend.get_chain(flat=False)
                chain_size = chain.shape[0]
                burnin = int(self.burnin_frac * chain_size)
                chain = backend.get_chain(flat=True, discard=burnin)
                logprob = backend.get_log_prob(flat=True, discard=burnin)
                if i == 0:
                    final_chain = chain
                    final_logprob = logprob
                else:
                    final_chain = np.vstack((final_chain, chain))
                    final_logprob = np.hstack((final_logprob, logprob))
        else:
            name = os.path.join(chain_dir, f'{handle}.h5')
            backend = emcee.backends.HDFBackend(name, read_only=True)
            chain = backend.get_chain(flat=False)
            chain_size = chain.shape[0]
            burnin = int(self.burnin_frac * chain_size)
            final_chain = backend.get_chain(flat=True, discard=burnin)
            final_logprob = backend.get_log_prob(flat=True, discard=burnin)

        index_min = np.argmax(final_logprob)
        ML_params = final_chain[index_min]
        return ML_params

    def get_logprob(self, handle: str, gelman: Optional[Dict] = None) -> np.ndarray:
        """
        Retrieve the log-probabilities from the MCMC chain.

        Args:
            handle (str): Handle used in the MCMC analysis.
            gelman (Dict, optional): Dictionary used as input for the Gelman-Rubin convergence 
            criteria. Defaults to None.

        Returns:
            np.ndarray: An array with the log-probabilities.
        """
        chain_dir = os.path.join(os.getcwd(), 'chains')

        if gelman is not None:
            N_chains = gelman['N']
            for i in range(N_chains):
                name = os.path.join(chain_dir, f'{handle}Run_{i}.h5')
                backend = emcee.backends.HDFBackend(name, read_only=True)
                chain = backend.get_chain(flat=False)
                chain_size = chain.shape[0]
                burnin = int(self.burnin_frac * chain_size)
                logprob = backend.get_log_prob(flat=True, discard=burnin)
                if i == 0:
                    final_logprob = logprob
                else:
                    final_logprob = np.hstack((final_logprob, logprob))
        else:
            name = os.path.join(chain_dir, f'{handle}.h5')
            backend = emcee.backends.HDFBackend(name, read_only=True)
            chain = backend.get_chain(flat=False)
            chain_size = chain.shape[0]
            burnin = int(self.burnin_frac * chain_size)
            final_logprob = backend.get_log_prob(flat=True, discard=burnin)

        return final_logprob


def main():
    prior = 'PRIOR_A_1T_stoc_hd_coev'    
    nwalkers = 30
    MCMC_test = MCMC(nwalkers,prior)
    MCMC_flat_prior = MCMC_test.in_prior
    gaussian_prior = MCMC_test.log_gaussian_prior
    x0 = [2.14]
    sigma = [0.2]
    print(gaussian_prior([2.14,2,2,2,2,2,2,2],x0=x0,sigma=sigma,params=['A']))



if __name__ == '__main__':
    main()



def try_mkdir(name: str) -> None:
    """
    Tries to create a directory with the given name in the current working directory.
    
    Parameters:
    name (str): The name of the directory to create.
    
    Returns:
    None
    """
    this_dir = os.getcwd()
    this_dir_to_make = os.path.join(this_dir, name)
    
    try:
        os.mkdir(this_dir_to_make)
    except FileExistsError:
        # Directory already exists, no action needed
        pass
    except OSError as e:
        # Handle other OS-related errors
        print(f"Error creating directory {this_dir_to_make}: {e}")