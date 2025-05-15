import numpy as np
from mcfit import P2xi, xi2P
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import sys 
from data_handling import read_bingo_results, deltaPfeature_bump, create_interpolation_function_bump, deltaPfeature_cpsc, create_interpolation_function_cpsc
try:
    from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
    from velocileptors.EPT.ept_fullresum_fftw import REPT

except ImportError:
    print("velocileptor package not found. Please install it to use the LPT model.")

class PowerSpectrumConstructor:
    def __init__(self, kh_array, ps_style = 'compressed', pf_model = None, ps_filename = None, 
                 transfer_function = None, norm = 0.05, h = 0.6736, fz = None, kIR = 0.2):
        """
        Load the file with the products necessary to evaluate the theoretical power spectrum.

        Args:
            ps_filename (str): Name of the file containing the momentum array, linear power spectrum,
                               smoothed linear power spectrum, and their ratio (O_lin).
            pf_model (str): Name of the pf_model to be used. The current options are:
                - 'lin'
                - 'log'
                - 'step'
                - 'sound'
                #Matteo's contribution
                - 'bump'
                - 'cpsc'
                - 'None' (i.e., usual BAO analysis without primordial features)
        
            ps_style (str): The style of the power spectrum. Default is 'compressed'.
                - 'compressed': Use the compressed power spectrum as in BOSS + eBOSS work
                - 'lpt_fixed': Use the LPT power spectrum (velocileptors) with transfer function fixed
                - 'lpt': Use the LPT power spectrum (velocileptors) with transfer function free (TODO)
                - 'ept_fixed': Use the EPT power spectrum (velocileptors)

            transfer_function (str): The transfer function to be used. Default is None. Should be provided if ps_style is 'lpt_fixed'            
        Raises:
            ValueError: If the pf_model is not one of the valid options.
        """
        # Constant used to normalize the broadband terms
        self.h = h
        self.k_norm = norm #[1/Mpc]
        self.kh_norm = norm / self.h  #[h/Mpc]

        # k array for the window function convolution
        self.kh_ext = np.logspace(-4, np.log10(10), 2**14) #[h/Mpc]
        self.k_ext = self.kh_ext * self.h  # [1/Mpc]

        #k Array to evaluate the theory (same as data)
        self.kh = np.array(kh_array) #input is h/Mpc
        self.k = self.kh*self.h #[1/Mpc]

        # Implemented feature pf_models
        pf_valid_options = {'lin', 'log', 'sound', 'step', 'bump','cpsc','None','external'}
        ps_style_options = {'compressed', 'lpt_fixed', 'lpt','ept_fixed'}

        if pf_model not in pf_valid_options:
            raise ValueError(f"Invalid pf_model '{pf_model}'. Please choose one of: {', '.join(pf_valid_options)}")
        self.pf_model = pf_model

        if ps_style not in ps_style_options:
            raise ValueError(f"Invalid ps_style '{ps_style}'. Please choose one of: {', '.join(ps_style_options)}")
        self.ps_style = ps_style

        if ps_style == "compressed":
            if ps_filename is None:
                raise ValueError("power spectrum for the compressed power spectrum must be provided.")
            self.initialize_compressed(ps_filename)
            self.Evaluate_bare = self.Evaluate_bare_compressed
            self.Evaluate_winconv = self.Evaluate_winconv_compressed

        if ps_style == 'lpt_fixed':
            if transfer_function is None:
                raise ValueError("Transfer function must be provided for LPT model.")
            self.initialize_lpt_fixed(transfer_function, fz, kIR)
            self.Evaluate_bare = self.Evaluate_bare_lpt_fixed
            self.Evaluate_winconv = None

        if ps_style == 'lpt_free':
            raise ValueError("Compressed power spectrum style is not implemented yet.")
            sys.exit(-1)

        if ps_style == 'ept_fixed':
            if ps_filename is None:
                raise ValueError("Need to provide a decomposed power spectrum for the EPT model.")
                sys.exit(-1)
            self.initialize_ept_fixed(ps_filename, fz)
            self.Evaluate_bare = self.Evaluate_bare_ept_fixed
            self.Evaluate_winconv = None

        # Map for different PF models of primordial features
        self.PrimordialFeatureModels = {
            'lin': self.LinearFeatures_deltaP,
            'log': self.LogarithmicFeatures_deltaP,
            'sound': self.VaryingSpeedOfSound_deltaP,
            'step': self.StepInPotential_deltaP,
            'None': lambda _: 0,
            #Matteo
            'bump': self.bump_deltaP,
            'cpsc': self.CPSC_deltaP,
        }

        if self.pf_model == 'cpsc':
            self.initialize_pf_cpsc()
        
        if self.pf_model == 'bump':
            self.initialize_pf_bump()
            
    def initialize_ept_fixed(self, ps_filename, fz):
        try:
            power_spec = np.loadtxt(ps_filename)
        except:
            raise ValueError(f"Problem loading the file: {ps_filename}")
            sys.exit(-1)

        kh_long, ps_smooth_points, O_lin_points, _ = power_spec.T

        # Store the values in the object
        kh_long = kh_long        
        self.ps_smooth = InterpolatedUnivariateSpline(kh_long, ps_smooth_points, ext=3)
        self.O_lin = InterpolatedUnivariateSpline(kh_long, O_lin_points, ext=3)
        self.fz = fz
    def initialize_pf_cpsc(self):
        filename = 'cosmologies/bingo_results_cpsc.h5'

        # Read data from the file
        k_values, log10_m_over_H_values, deltaP_values = read_bingo_results(filename)

        # Create the interpolation function
        interp_func = create_interpolation_function_cpsc(k_values, log10_m_over_H_values, deltaP_values)

        self.CPSC_interp = interp_func

    def initialize_pf_bump(self):
        filename = 'cosmologies/bingo_results_bump.h5'

        # Read data from the file
        k_values, deltaN, deltaP_values = read_bingo_results(filename)

        # Create the interpolation function
        interp_func = create_interpolation_function_bump(k_values, deltaN, deltaP_values)

        self.bump_interp = interp_func

    def initialize_compressed(self, ps_filename):
        """
        Load the file with the products necessary to evaluate the theoretical power spectrum.

        Args:
            ps_filename (str): Name of the file containing the momentum array, linear power spectrum,
                               smoothed linear power spectrum, and their ratio (O_lin).
        """
        try:
            power_spec = np.loadtxt(ps_filename)
        except:
            raise ValueError(f"Problem loading the file: {ps_filename}")
            sys.exit(-1)

        kh_long, ps_smooth_points, O_lin_points, ps_lin_points = power_spec.T

        # Store the values in the object
        self.kh_long = kh_long        
        self.ps_smooth = InterpolatedUnivariateSpline(kh_long, ps_smooth_points, ext=3)
        self.O_lin = InterpolatedUnivariateSpline(kh_long, O_lin_points, ext=3)
        self.ps_lin = InterpolatedUnivariateSpline(kh_long, ps_lin_points, ext=3)
        self.P_norm = 1e3

    def initialize_lpt_fixed(self, transfer_function, fz, kIR):
        """
        Load the transfer function for the LPT model.

        Args:
            transfer_function (str): The transfer function to be used.
            fz (float): The redshift factor.
            kIR (float): The IR cutoff scale.
        """
        try:
            k, transfer = np.loadtxt(transfer_function, unpack=True)
            transfer_func = interp1d(k, transfer)
            self.transfer2_plin = transfer_func(self.k)**2*self.k**4 / (self.k**3/(2*np.pi**2))
            self.transfer2_plin_ext = transfer_func(self.k_ext)**2*self.k_ext**4 / (self.k_ext**3/(2*np.pi**2))
            self.fz = fz
            self.kIR = kIR
            
        except:
            raise ValueError(f"Problem loading the file: {transfer_function}")
            sys.exit(-1)

    def DefineWindowFunction(self, winfunc):
        """
        Initialise the survey window function.
        Args:
            winfunc (Callable[[float], float]): The interpolated function representing the 
            window function in configuration space.
        """
        self.winfunc = winfunc

    def ApplyWindowFunction(self, P_bare, window_func):
        """
        Apply a window function to the power spectrum.

        Args:
            P_bare (array-like): The bare power spectrum.
            window_func (Callable[[float], float]): The window function to be applied.

        Returns:
            InterpolatedUnivariateSpline: The power spectrum after applying the window function.
        """
        # Put the theory in configuration space
        s, xi_bare = P2xi(self.kh_ext)(P_bare)

        # Apply the window function
        xi = xi_bare * window_func(s)

        # Put the theory back in Fourier Space
        kh_output, P_window = xi2P(s)(xi)

        return InterpolatedUnivariateSpline(kh_output, P_window)

    def SmoothAmplitude(self, _k, sigma_s, B2, a0, a1, a2, a3, a4):
        """
        Compute the smooth amplitude for the power spectrum.

        Args:
            sigma_s (float): Velocity damping term.
            B2 (float): Amplitude parameter.
            a0, a1, a2, a3, a4 (float): Broadband parameters.

        Returns:
            array-like: The smooth power spectrum.
        """
        # Compute the velocity damping term
        F_fog = 1.0 / (1.0 + 0.5 * _k**2 * sigma_s**2)**2

        # Theory - Broadband
        invk_norm = self.kh_norm / _k
        theory_broadband = (
            a0 * invk_norm**3 + a1 * invk_norm**2 + a2 * invk_norm + a3 +
            (a4 * (_k)**2) * np.exp(-0.1 * _k**2)
        )

        # Compute the full non-wiggle part
        P_nw = B2**2 * self.ps_smooth(_k) * F_fog + theory_broadband * self.P_norm

        return P_nw

    def VaryingSpeedOfSound_deltaP(self, _k, params):
        """
        Compute the delta power spectrum for the 'sound' model.

        Args:
            params (list): List of parameters [tau_f, log_beta, As].

        Returns:
            array-like: The delta power spectrum.
        """
        # Unpack the primordial features parameters
        As, tau_f, log_beta = params

        tau_f = -tau_f * self.h
        kd = np.exp(log_beta)**0.5 / tau_f
        osc1 = np.sin(2 * tau_f * _k) + 1 / (tau_f * _k) * np.cos(2 * tau_f * _k)
        osc2 = -0.5 / (tau_f**2 * _k) * np.sin(2 * tau_f * _k)
        Dk = As * np.exp(-_k**2 / kd**2) * _k * np.sqrt(np.pi) / (9 * kd)
        d_Dk = (As / (9 * kd**3)) * np.exp(-_k**2 / kd**2) * (kd**2 - 2 * _k**2) * np.sqrt(np.pi)
        delta_P = osc1 * Dk + osc2 * d_Dk
        return delta_P

    def LinearFeatures_deltaP(self, _k, params):
        """
        Compute the delta power spectrum for the 'lin' model.

        Args:
            params (list): List of parameters [A, omega_lin, phi].

        Returns:
            array-like: The delta power spectrum.
        """
        # Unpack the primordial features parameters
        A, omega_lin, phi = params

        # Linear Primordial Oscillations
        delta_P = A * np.sin(omega_lin * _k * self.h + np.pi * phi)
        return delta_P

    def LogarithmicFeatures_deltaP(self, _k, params):
        """
        Compute the delta power spectrum for the 'log' model.

        Args:
            params (list): List of parameters [A, omega_log, phi].

        Returns:
            array-like: The delta power spectrum.
        """
        # Unpack the primordial features parameters
        A, omega_log, phi = params

        # Logarithmic Primordial Oscillations
        delta_P = A * np.sin(omega_log * np.log(_k / self.k_norm) + np.pi * phi)
        return delta_P

    def StepInPotential_deltaP(self, _k, params):
        """
        Compute the delta power spectrum for the 'step' model.

        Args:
            params (list): List of parameters [As, omegas, xs].

        Returns:
            array-like: The delta power spectrum.
        """
        # Unpack the primordial features parameters
        As, omegas, xs = params

        x = _k * omegas
        D_arg = x / xs
        W0 = 1 / (2 * x**4) * ((18 * x - 6 * x**3) * np.cos(2 * x) + (15 * x**2 - 9) * np.sin(2 * x))
        W1 = -3 / (x**4) * (x * np.cos(x) - np.sin(x)) * (3 * x * np.cos(x) + (2 * x**2 - 3) * np.sin(x))
        D = D_arg / np.sinh(D_arg)
        I0 = As * W0 * D
        I1 = 1 / np.sqrt(2) * (np.pi / 2 * (1 - self.ns) + As * W1 * D)
        deltaI0 = np.exp(I0) - 1

        # The total feature contribution to the wiggly power spectrum
        delta_P = deltaI0 + I1**2 + I1**2 * deltaI0
        return delta_P

    #Matteo
    def BumpInPotential_deltaP(self, _k, params):
        """
        Compute the delta power spectrum for the 'bump' model.
        The calculation is performed numerically by solving the MS equation using the code BINGO, then interpolated.

        Args:
            params (list): List of parameters [deltaN, N0, dP].

        Returns:
            array-like: The delta power spectrum.
        """
        filename = 'cosmologies/bingo_results_bump.h5'
    
        # Read data from the file
        k_values, deltaN_values, deltaP_values = read_bingo_results(filename)
    
        # Create the interpolation function
        interp_func = create_interpolation_function_bump(k_values, deltaN_values, deltaP_values)
        
        # Unpack the primordial features parameters
        dP, N0, deltaN = params

        return deltaPfeature_bump(_k  * self.h,  dP, N0, deltaN, interp_func)
    
    def CPSC_deltaP(self, _k, params):
        """
        Compute the delta power spectrum for the 'cpsc' model.
        The calculation is performed numerically by solving the MS equation using the code BINGO, then interpolated.

        Args:
            params (list): List of parameters [log10_m_ver_h, N0, dP].

        Returns:
            array-like: The delta power spectrum.
        """
        
        # Unpack the primordial features parameters
        dP, N0, log10_m_over_H = params

        # Amplitude modulation term
        Amp = dP / 0.01
        
        arg = _k / np.exp(N0-0.135036080469101E+002) * self.h

        # Interpolate deltaP using the interpolation function
        deltaP = Amp*self.CPSC_interp(np.log10(arg), log10_m_over_H, grid=False)
        return deltaP
    
    def bump_deltaP(self,_k,params):
        """
        Compute the delta power spectrum for the 'bump' model.
        The calculation is performed numerically by solving the MS equation using the code BINGO, then interpolated.

        Args:
            params (list): List of parameters [dP, N0, deltaN].

        Returns:
            array-like: The delta power spectrum.
        """
                
        # Unpack the primordial features parameters
        dP, N0, deltaN = params

        # Amplitude modulation term
        Amp = dP / 0.025
        
        arg = _k / np.exp(N0-15)

        # Interpolate deltaP using the interpolation function
        deltaP = Amp*self.bump_interp(np.log10(arg), np.log10(deltaN), grid=False)
        
        # Calculate the power spectrum with the feature
        return deltaP
    
#########################################################################################################    
    def BAO(self, _k, alpha):
        """
        Compute the BAO oscillations.

        Args:
            alpha (float): Scaling parameter for the BAO oscillations.

        Returns:
            array-like: The BAO oscillations.
        """
        # BAO oscillations
        O_lin_points = self.O_lin(_k / alpha) - 1
        return O_lin_points

    def NonlinearDamping(self, _k, sigma_nl):
        """
        Compute the nonlinear damping factor.

        Args:
            sigma_nl (float): Nonlinear damping scale.

        Returns:
            array-like: The nonlinear damping factor.
        """
        return np.exp(-0.5 * _k**2 * sigma_nl**2)

    def Evaluate_bare_ept_fixed(self, params):
            """
            Evaluate the bare power spectrum using the LPT with fixed transfer function

            Args:
                params (list): List of parameters [b1, b2, bs, b3, alpha0, alpha2, alpha4,
                shot0, shot2, *deltaP_params].

            Returns:
                array-like: The evaluated power spectrum.
            """ 

            # Get the broadband + feature parameters
            b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, shot0, shot2, shot4, *deltaP_params = params

            biases = [b1, b2, bs, b3]
            cterms = [alpha0, alpha2, alpha4, alpha6]
            stoch = [shot0, shot2, shot4]
            pars = biases + cterms + stoch

            # Compute delta_P (primordial feature)

            if deltaP_params == []:
                deltaP = self.PrimordialFeatureModels[self.pf_model](self.kh)
            else:
                deltaP = self.PrimordialFeatureModels[self.pf_model](self.kh, deltaP_params)

            # BAO oscillations
            BAO_oscillations = self.O_lin(self.kh) - 1
            pnw = self.ps_smooth(self.kh)
            plin = pnw * (1.0 + deltaP + deltaP * BAO_oscillations + BAO_oscillations)

            ept = REPT(self.kh,plin, pnw=pnw, kmin = 5e-3, kmax = 0.25, nk = 1000,\
            beyond_gauss=True, one_loop= True,\
            N = 5000, extrap_min=-6, extrap_max=4, cutoff = 100, threads=1)
            ks, p0, p2, p4 = ept.compute_redshift_space_power_multipoles(pars,self.fz)
            p0 = interp1d(ks,p0)(self.kh)

            return p0,plin,pnw
    
    def Evaluate_bare_lpt_fixed(self, params):
            """
            Evaluate the bare power spectrum using the LPT with fixed transfer function

            Args:
                params (list): List of parameters [As, ns, b1, b2, bs, b3, alpha0, alpha2, alpha4,
                shot0, shot2, *deltaP_params].

            Returns:
                array-like: The evaluated power spectrum.
            """ 
            # Get the broadband + feature parameters
            As, ns, b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, shot0, shot2, shot4, *deltaP_params = params

            biases = [b1, b2, bs, b3]
            cterms = [alpha0, alpha2, alpha4, alpha6]
            stoch = [shot0, shot2, shot4]
            pars = biases + cterms + stoch

            # Compute delta_P (primordial feature)

            if deltaP_params == []:
                deltaP = self.PrimordialFeatureModels[self.pf_model](self.kh)
            else:
                deltaP = self.PrimordialFeatureModels[self.pf_model](self.kh, deltaP_params)

            plin = As * (self.k/self.k_norm)**(ns - 1)*(1.0 + deltaP) * self.transfer2_plin * self.h**3
            lpt = LPT_RSD(self.kh,plin,kIR=self.kIR)
            lpt.make_pltable(self.fz,nmax=6,apar=1,aperp=1, kv = self.kh)
            kl,p0,p2,p4 = lpt.combine_bias_terms_pkell(pars)

            return kl, p0, plin
    
    def Evaluate_bare_compressed(self, params):
        """
        Evaluate the bare power spectrum without applying the window function.

        Args:
            kh_data (array-like): Array of k-values at which to evaluate the power spectrum.
            params (list): List of parameters [BNGC, sigma_nl, sigma_s, a0, a1, a2, a3, a4, 
            alpha, *deltaP_params].

        Returns:
            array-like: The evaluated power spectrum.
        """
        # Get the broadband + feature parameters
        B, a0, a1, a2, a3, a4, alpha, sigma_nl, sigma_s, *deltaP_params = params
        # Compute delta_P (primordial feature)

        if deltaP_params == []:
            deltaP = self.PrimordialFeatureModels[self.pf_model](self.kh)
        else:
            deltaP = self.PrimordialFeatureModels[self.pf_model](self.kh, deltaP_params)

        # Get the smooth power spectrum for NGC and SGC 
        P_nw = self.SmoothAmplitude(self.kh, sigma_s, B, a0, a1, a2, a3, a4)

        # BAO oscillations
        BAO_wiggles = self.BAO(self.kh, alpha)

        # Nonlinear Damping
        nonlinear_damping = self.NonlinearDamping(self.kh, sigma_nl)

        # Final Result
        P0_bare = P_nw * (1 + (BAO_wiggles + deltaP + deltaP * BAO_wiggles) * nonlinear_damping)

        return P0_bare
        
    def Evaluate_winconv_compressed(self, params):
        """
        Evaluate the power spectrum with window function convolution.

        Args:
            kh_data (array-like): Array of k-values at which to evaluate the power spectrum.
            params (list): List of parameters [B, sigma_nl, sigma_s, a0, a1, a2, a3, a4, 
            alpha, *deltaP_params].

        Returns:
            array-like: The evaluated power spectrum with window function convolution.
        """
        # Get the broadband + feature parameters
        B, a0, a1, a2, a3, a4, alpha, sigma_nl, sigma_s, *deltaP_params = params
        
        if deltaP_params == []:
            deltaP = self.PrimordialFeatureModels[self.pf_model](self.kh_ext)
        else:
            deltaP = self.PrimordialFeatureModels[self.pf_model](self.kh_ext, deltaP_params)

        # Get the smooth power spectrum for NGC and SGC 
        P_nw = self.SmoothAmplitude(self.kh_ext, sigma_s, B, a0, a1, a2, a3, a4)

        # BAO oscillations
        BAO_wiggles = self.BAO(self.kh_ext, alpha)

        # Nonlinear Damping
        nonlinear_damping = self.NonlinearDamping(self.kh_ext, sigma_nl)

        # Final Result
        P0_bare = P_nw * (1 + (BAO_wiggles + deltaP + deltaP * BAO_wiggles) * nonlinear_damping)

        # Apply window function
        P0 = self.ApplyWindowFunction(P0_bare, self.winfunc)(self.kh)

        return P0