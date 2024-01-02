#!/usr/bin/env python

# the purpose of this script is to fit glm-hmm to mouse choice data
# and output the posterior probability which is glm-hmm state probability 
# into a csv file for subsequent protting or other data manipulations

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import ssm
import psytrack as psy
import copy
from ssm.util import find_permutation
import argparse
from collections import defaultdict
import copy
import math
from matplotlib.gridspec import GridSpec


def get_args():

    """This grabs arguments for setting total trials"""

    parser = argparse.ArgumentParser(description="global variables to set input file names")
    parser.add_argument("-ch", "--choices", help="choices file name", required=True, type=str)
    parser.add_argument("-ip", "--inputs", help="input file name", required=True, type=str)
    #parser.add_argument("-rt", "--react_times", help="reaction times file name", required=True, type=str)
    parser.add_argument("-g", "--gname", help="graph name", required=True, type=str)
    return parser.parse_args()


def fit_glm_hmm(hmm, filt_choices, filt_inpts, N_iters,TOL):

    """Takes newly initialized hmm glm model (without standard transition matrix and weights) and 
    returns a fitted glm hmm. Returns fitted glm-hmm"""

    fit_glmhmm = hmm.fit(np.concatenate(filt_choices), inputs=np.concatenate(filt_inpts), method="em",
                          num_iters=N_iters, tolerance=TOL)
    
    return fit_glmhmm


if __name__ == "__main__":
    args = get_args()
    choice_f: str = args.choices #holds path or file name of mouse choice data
    input_f: str = args.inputs #holds path or file name of mouse inputs data
    experiment: str = args.gname #holds experiment number to use as a graph header
         #load mouse inputs and choices
    npr.seed(42)

    inputs = np.load(input_f,allow_pickle = True) # load numpy file inputs
    choices = np.load(choice_f,allow_pickle = True) # load numpy files choices

    input_dim: int = inputs[0].shape[1]  # input dimensions
    num_states = 3        # number of discrete states
    TOL: float = 10**-4 # tolerance 
    N_iters: int = 1000 # number of iterations for the fitting model

    obs_dim: int = choices[0].shape[1]          # number of observed dimensions
    num_categories: int = len(np.unique(np.concatenate(choices)))    # number of categories for output

    hmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")
    fit_glmhmm = fit_glm_hmm(hmm, choices, inputs, N_iters,TOL)