import numpy as np
from mcfit import P2xi, xi2P
from scipy.interpolate import InterpolatedUnivariateSpline
import sys 
from data_handling import read_bingo_results, deltaPfeature_bump, create_interpolation_function_bump, deltaPfeature_cpsc, create_interpolation_function_cpsc
class PowerSpectrumConstructor:
    def __init__(self, ps_filename, model, k_array):
        """
        Load the file with the products necessary to evaluate the theoretical power spectrum.

        Args:
            ps_filename (str): Name of the file containing the momentum array, linear power spectrum,
                               smoothed linear power spectrum, and their ratio (O_lin).
            model (str): Name of the model to be used. The current options are:
                - 'lin'
                - 'log'
                - 'step'
                - 'sound'
                #Matteo's contribution
                - 'bump'
                - 'cpsc'
                - 'None' (i.e., usual BAO analysis without primordial features)
        
        Raises:
            ValueError: If the model is not one of the valid options.
        """
        
        # Load the file and unpack the data
        try:
            power_spec = np.loadtxt(ps_filename)
        except:
            print("Problem loading the file:", ps_filename)
            sys.exit(-1)

        # Implemented feature models
        valid_options = {'lin', 'log', 'sound', 'step', 'bump','cpsc','None','external'}

        if model not in valid_options:
            raise ValueError(f"Invalid model '{model}'. Please choose one of: {', '.join(valid_options)}")

        kh_long, ps_smooth_points, O_lin_points, ps_lin_points = power_spec.T
        
        # Store the values in the object
        self.kh_long = kh_long        
        self.ps_smooth = InterpolatedUnivariateSpline(kh_long, ps_smooth_points, ext=3)
        self.O_lin = InterpolatedUnivariateSpline(kh_long, O_lin_points, ext=3)
        self.ps_lin = InterpolatedUnivariateSpline(kh_long, ps_lin_points, ext=3)
        self.model = model

        # Constant used to normalize the broadband terms
        self.P_norm = 1e3  # [Mpc/h]^3

        # Fiducial parameters TODO: change the way it is given as input
        self.h = 0.6736
        self.ns = 0.96

        # Normalization for the broadband terms
        self.k_norm = 0.05 / self.h  # [h/Mpc]

        # Array for the Hankel transform
        self.kh_ext = np.logspace(-4, np.log10(10), 2**14)

        # Array to evaluate the theory
        self.k = k_array

        # Map for different models of primordial features
        self.PrimordialFeatureModels = {
            'lin': self.LinearFeatures_deltaP,
            'log': self.LogarithmicFeatures_deltaP,
            'sound': self.VaryingSpeedOfSound_deltaP,
            'step': self.StepInPotential_deltaP,
            #Matteo
            'bump': self.BumpInPotential_deltaP,
            'cpsc': self.CPSC_deltaP,
            'None': lambda _: 0,
        }
    
    def external_deltaP(self, func):
        self.PrimordialFeatureModels['external'] = func

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
        invk_norm = self.k_norm / _k
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
        tau_f, log_beta, As = params

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
            params (list): List of parameters [omegas, xs, As].

        Returns:
            array-like: The delta power spectrum.
        """
        # Unpack the primordial features parameters
        omegas, xs, As = params

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
        filename = 'cosmologies/bingo_results_cpsc.h5'
    
        # Read data from the file
        k_values, log10_m_over_H_values, deltaP_values = read_bingo_results(filename)
    
        # Create the interpolation function
        interp_func = create_interpolation_function_cpsc(k_values, log10_m_over_H_values, deltaP_values)

        # Unpack the primordial features parameters
        dP, N0, log10_m_over_H = params

        return deltaPfeature_cpsc(_k  * self.h,  dP, N0, log10_m_over_H, interp_func)
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

    def Evaluate_bare(self, params):
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
            deltaP = self.PrimordialFeatureModels[self.model](self.k)
        else:
            deltaP = self.PrimordialFeatureModels[self.model](self.k, deltaP_params)

        # Get the smooth power spectrum for NGC and SGC 
        P_nw = self.SmoothAmplitude(self.k, sigma_s, B, a0, a1, a2, a3, a4)

        # BAO oscillations
        BAO_wiggles = self.BAO(self.k, alpha)

        # Nonlinear Damping
        nonlinear_damping = self.NonlinearDamping(self.k, sigma_nl)

        # Final Result
        P0_bare = P_nw * (1 + (BAO_wiggles + deltaP + deltaP * BAO_wiggles) * nonlinear_damping)

        return P0_bare
        
    def Evaluate_wincov(self, params):
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
            deltaP = self.PrimordialFeatureModels[self.model](self.kh_ext)
        else:
            deltaP = self.PrimordialFeatureModels[self.model](self.kh_ext, deltaP_params)

        # Get the smooth power spectrum for NGC and SGC 
        P_nw = self.SmoothAmplitude(self.kh_ext, sigma_s, B, a0, a1, a2, a3, a4)

        # BAO oscillations
        BAO_wiggles = self.BAO(self.kh_ext, alpha)

        # Nonlinear Damping
        nonlinear_damping = self.NonlinearDamping(self.kh_ext, sigma_nl)

        # Final Result
        P0_bare = P_nw * (1 + (BAO_wiggles + deltaP + deltaP * BAO_wiggles) * nonlinear_damping)

        # Apply window function
        P0 = self.ApplyWindowFunction(P0_bare, self.winfunc)(self.k)

        return P0