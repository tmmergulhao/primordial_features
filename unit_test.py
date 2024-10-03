# main.py
import numpy as np
import ps_constructor
import likelihood
import mcmc_toolkit
import pypower
import time
import os
import argparse
import logging
import json
import sys
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from multiprocessing import Pool
from dotenv import load_dotenv
import data_handling
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

#Load the data
DATA = np.loadtxt('/Users/s2223060/Desktop/primordial_features/data/BOSS/gal/pk/pk_BOSS_NGC_highZ_dk_0.001.txt')
covariance = np.loadtxt('/Users/s2223060/Desktop/primordial_features/data/BOSS/gal/cov/COV_BOSS_NGC_highZ_dk_0.001.txt')
k = np.loadtxt('/Users/s2223060/Desktop/primordial_features/data/BOSS/gal/k_BOSS_NGC_highZ_dk_0.001.txt')


PLIN = '/Users/s2223060/Desktop/primordial_features/cosmologies/power_spectrum/BOSS_z3_new.txt'
primordialfeature_model = 'lin'

wfunc_NGC = np.loadtxt('/Users/s2223060/Desktop/primordial_features/data/BOSS/gal/PairCount_W(s)_win_CARTESIAN_0.5_0.75_random0_DR12v5_CMASSLOWZTOT_North.txt')
wfunc_SGC = np.loadtxt('/Users/s2223060/Desktop/primordial_features/data/BOSS/gal/PairCount_W(s)_win_CARTESIAN_0.5_0.75_random0_DR12v5_CMASSLOWZTOT_South.txt')

wfunc_NGC = InterpolatedUnivariateSpline(wfunc_NGC.T[0],wfunc_NGC.T[1]/wfunc_NGC.T[1][0], ext = 3)
wfunc_SGC = InterpolatedUnivariateSpline(wfunc_SGC.T[0],wfunc_SGC.T[1]/wfunc_SGC.T[1][0], ext = 3)

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
ps_model_NGC.DefineWindowFunction(wfunc_NGC)
theory_NGC = lambda x: ps_model_NGC.Evaluate_wincov(x)


# Initialize the model for SGC
ps_model_SGC = ps_constructor.PowerSpectrumConstructor(PLIN, primordialfeature_model, k)
ps_model_SGC.DefineWindowFunction(wfunc_SGC)
theory_SGC = lambda x: ps_model_SGC.Evaluate_wincov(x)

#Create the theory for the whole data space
def theory(theta):
    theta_NGC = [theta[0],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7],theta[8],theta[9],theta[10],theta[11],theta[12]]
    theta_SGC = [theta[1],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7],theta[8],theta[9],theta[10],theta[11],theta[12]]
    return np.hstack((theory_NGC(theta_NGC),theory_SGC(theta_SGC)))

#***************************************************************************************************
invcov = np.linalg.inv(covariance)

#Create the likelihood
PrimordialFeature_likelihood = likelihood.likelihoods(theory, DATA, invcov)

unit_test = np.loadtxt('/Users/s2223060/Desktop/primordial_features/priors/unit_test_lin_range1.txt')

output = []
for theta in tqdm(unit_test):
    output.append(PrimordialFeature_likelihood.logGaussian(theta))
np.savetxt('/Users/s2223060/Desktop/primordial_features/priors/unit_test_logprobs.txt',output)