import numpy as np
from mcfit import P2xi, xi2P
from scipy.interpolate import InterpolatedUnivariateSpline
import sys

class PowerSpectrumConstructor:
    def __init__(self, ps_filename, model):
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
        valid_options = {'lin', 'log', 'sound', 'step', 'None'}

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
        self.P_norm = 1e6

        # Fiducial parameters TODO: change the way it is given as input
        self.h = 0.676
        self.ns = 0.96

        # Normalization for the broadband terms
        self.k_norm = 0.05 / self.h  # [h/Mpc]

        # Array for the Hankel transform
        self.kh_ext = np.logspace(-4, np.log10(10), 2**12)

        # Map for different models of primordial features
        self.PrimordialFeatureModels = {
            'lin': self.LinearFeatures_deltaP,
            'log': self.LogarithmicFeatures_deltaP,
            'sound': self.VaryingSpeedOfSound_deltaP,
            'step': self.StepInPotential_deltaP,
            'None': lambda _: 0
        }
    
    def DefineWindowFunction(self, winfunc_NGC, winfunc_SGC):
        """
        Initialise the survey window function for both NGC and SGC.

        Args:
            winfunc_NGC (Callable[[float], float]): The interpolated function representing the 
                                                    window function for the NGC in configuration space.
            winfunc_SGC (Callable[[float], float]): The interpolated function representing the 
                                                    window function for the SGC in configuration space.
        """
        self.winfunc_NGC = winfunc_NGC
        self.winfunc_SGC = winfunc_SGC

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

    def SmoothAmplitude(self, sigma_s, B2, a0, a1, a2, a3, a4):
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
        F_fog = 1.0 / (1.0 + 0.5 * self.kh_ext**2 * sigma_s**2)**2

        # Theory - Broadband
        invk_norm = self.k_norm / self.kh_ext
        theory_broadband = (
            a0 * invk_norm**3 + a1 * invk_norm**2 + a2 * invk_norm + a3 +
            a4 * (self.kh_ext)**2 * np.exp(-0.1 * self.kh_ext**2)
        )

        # Compute the full non-wiggle part
        P_nw = B2**2 * self.ps_smooth(self.kh_ext) * F_fog + theory_broadband * self.P_norm

        return P_nw

    def VaryingSpeedOfSound_deltaP(self, params):
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
        osc1 = np.sin(2 * tau_f * self.kh_ext) + 1 / (tau_f * self.kh_ext) * np.cos(2 * tau_f * self.kh_ext)
        osc2 = -0.5 / (tau_f**2 * self.kh_ext) * np.sin(2 * tau_f * self.kh_ext)
        Dk = As * np.exp(-self.kh_ext**2 / kd**2) * self.kh_ext * np.sqrt(np.pi) / (9 * kd)
        d_Dk = (As / (9 * kd**3)) * np.exp(-self.kh_ext**2 / kd**2) * (kd**2 - 2 * self.kh_ext**2) * np.sqrt(np.pi)
        delta_P = osc1 * Dk + osc2 * d_Dk
        return delta_P

    def LinearFeatures_deltaP(self, params):
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
        delta_P = A * np.sin(omega_lin * self.kh_ext * self.h + np.pi * phi)
        return delta_P

    def LogarithmicFeatures_deltaP(self, params):
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
        delta_P = A * np.sin(omega_log * np.log(self.kh_ext / self.k_norm) + np.pi * phi)
        return delta_P

    def StepInPotential_deltaP(self, params):
        """
        Compute the delta power spectrum for the 'step' model.

        Args:
            params (list): List of parameters [omegas, xs, As].

        Returns:
            array-like: The delta power spectrum.
        """
        # Unpack the primordial features parameters
        omegas, xs, As = params

        x = self.kh_ext * omegas
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

    def BAO(self, alpha):
        """
        Compute the BAO oscillations.

        Args:
            alpha (float): Scaling parameter for the BAO oscillations.

        Returns:
            array-like: The BAO oscillations.
        """
        # BAO oscillations
        O_lin_points = self.O_lin(self.kh_ext / alpha) - 1
        return O_lin_points

    def NonlinearDamping(self, sigma_nl):
        """
        Compute the nonlinear damping factor.

        Args:
            sigma_nl (float): Nonlinear damping scale.

        Returns:
            array-like: The nonlinear damping factor.
        """
        return np.exp(-0.5 * self.kh_ext**2 * sigma_nl**2)

    def Evaluate_bare(self, kh_data, params):
        """
        Evaluate the bare power spectrum without applying the window function.

        Args:
            kh_data (array-like): Array of k-values at which to evaluate the power spectrum.
            params (list): List of parameters [BNGC, BSGC, sigma_nl, sigma_s, a0, a1, a2, a3, a4, 
            alpha, *deltaP_params].

        Returns:
            array-like: The evaluated power spectrum.
        """
        # Get the broadband + feature parameters
        BNGC, BSGC, sigma_nl, sigma_s, a0, a1, a2, a3, a4, alpha, *deltaP_params = params
        
        # Compute delta_P (primordial feature)
        deltaP = self.PrimordialFeatureModels[self.model](deltaP_params)

        # Get the smooth power spectrum for NGC and SGC 
        P_nw_NGC = self.SmoothAmplitude(sigma_s, BNGC, a0, a1, a2, a3, a4)
        P_nw_SGC = self.SmoothAmplitude(sigma_s, BSGC, a0, a1, a2, a3, a4)

        # BAO oscillations
        BAO_wiggles = self.BAO(alpha)

        # Nonlinear Damping
        nonlinear_damping = self.NonlinearDamping(sigma_nl)

        # Final Result
        P0_bare_NGC = P_nw_NGC * (1 + (BAO_wiggles + deltaP + deltaP * BAO_wiggles) * nonlinear_damping)
        P0_bare_SGC = P_nw_SGC * (1 + (BAO_wiggles + deltaP + deltaP * BAO_wiggles) * nonlinear_damping)

        P0_NGC = InterpolatedUnivariateSpline(self.kh_ext, P0_bare_NGC)
        P0_SGC = InterpolatedUnivariateSpline(self.kh_ext, P0_bare_SGC)

        return np.hstack((P0_NGC(kh_data), P0_SGC(kh_data)))
        
    def Evaluate_wincov(self, kh_data, params):
        """
        Evaluate the power spectrum with window function convolution.

        Args:
            kh_data (array-like): Array of k-values at which to evaluate the power spectrum.
            params (list): List of parameters [BNGC, BSGC, sigma_nl, sigma_s, a0, a1, a2, a3, a4, 
            alpha, *deltaP_params].

        Returns:
            array-like: The evaluated power spectrum with window function convolution.
        """
        # Get the broadband + feature parameters
        BNGC, BSGC, sigma_nl, sigma_s, a0, a1, a2, a3, a4, alpha, *deltaP_params = params
        
        # Compute delta_P (primordial feature)
        deltaP = self.PrimordialFeatureModels[self.model](deltaP_params)

        # Get the smooth power spectrum for NGC and SGC 
        P_nw_NGC = self.SmoothAmplitude(sigma_s, BNGC, a0, a1, a2, a3, a4)
        P_nw_SGC = self.SmoothAmplitude(sigma_s, BSGC, a0, a1, a2, a3, a4)

        # BAO oscillations
        BAO_wiggles = self.BAO(alpha)

        # Nonlinear Damping
        nonlinear_damping = self.NonlinearDamping(sigma_nl)

        # Final Result
        P0_bare_NGC = P_nw_NGC * (1 + (BAO_wiggles + deltaP + deltaP * BAO_wiggles) * nonlinear_damping)
        P0_bare_SGC = P_nw_SGC * (1 + (BAO_wiggles + deltaP + deltaP * BAO_wiggles) * nonlinear_damping)

        # Apply window function
        P0_NGC = self.ApplyWindowFunction(P0_bare_NGC, self.winfunc_NGC)
        P0_SGC = self.ApplyWindowFunction(P0_bare_SGC, self.winfunc_SGC)

        return np.hstack((P0_NGC(kh_data), P0_SGC(kh_data)))