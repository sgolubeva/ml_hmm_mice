#!/usr/bin/env python

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


def get_args():

    """This grabs arguments for setting total trials"""

    parser = argparse.ArgumentParser(description="global variables to set input file names")
    parser.add_argument("-ch", "--choices", help="choices file name", required=True, type=str)
    parser.add_argument("-ip", "--inputs", help="input file name", required=True, type=str)
    parser.add_argument("-g", "--gname", help="graph name", required=True, type=str)
    return parser.parse_args()


def filter_no_response_cho(inputs, choices, cho=2):

    """Takes numpy arrays of inputs and choices, filters out choice 2 (no response) from choice array
    and the corresponding indexes from the inputs array to keep the shapes of both the same. Returns
    an array of filtered out choices and filtered out inputs"""

    new_choices = list()
    new_inputs = list()
    for i in range(len(choices)):
        inds = np.squeeze(choices[i]!=cho)
        new_inputs.append(inputs[i][inds,:])
        new_choices.append(choices[i][inds])
    new_choices = np.array(new_choices, dtype='O')
    new_inputs = np.array(new_inputs, dtype='O')
    return new_inputs, new_choices


def fit_glm_hmm(hmm, filt_choices, filt_inpts, N_iters,TOL):

    """Takes newly initialized hmm glm model (without standard transition matrix and weights) and 
    returns a fitted glm hmm. Returns fitted glm-hmm"""

    # new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
    #                         observation_kwargs=dict(C=num_categories), transitions="standard")

    # fit_glmhmm = new_glmhmm.fit(np.concatenate(filt_choices), inputs=np.concatenate(filt_inpts),
    #                              method="em", num_iters=N_iters, tolerance=tolerance)

    fit_glmhmm = hmm.fit(np.concatenate(filt_choices), inputs=np.concatenate(filt_inpts), method="em",
                          num_iters=N_iters, tolerance=TOL)
    #import ipdb; ipdb.set_trace()
    return fit_glmhmm



##################################################################
# plot choices, and states
##################################################################

def plot_states(hmm, filt_choices, filt_inputs, num_states, sess_id = None):

    """Takes fitted glm hmm and an axis object for plotting and plots states on the second subplot"""
    
    cols = ['#ff7f00', '#4daf4a', '#377eb8']
    posterior_probs_new = [hmm.expected_states(data=data, input=inpt)[0]
                for data, inpt
                in zip(filt_choices, filt_inputs)] # plot states predicted by the model not initialized with statndard weights and matrices
    #posterior_probs_true = [true_glmhmm.expected_states(data=data, input=inpt)[0]
                #for data, inpt
                #in zip(true_choice, inpts)] # plot states predicted by the model initialized with standard weights and matrices
    
    # plot true states of the model when it generated the choice data
    # x_range = np.array(range(len(true_latents[0])))
    # for i in range(len(cols)):
    #     mask = (true_latents[0] == i)
    #     y_values = np.ones(len(true_latents[0][mask]))
    #     ax.scatter(x_range[mask], y_values*1.25, c=cols[i], label=f'state {i+1}')

    
    #sess_id = 0 #session id; can choose any index between 0 and num_sess-1
    plt.figure(figsize=(40,12))
    for k in range(num_states):
        plt.plot(posterior_probs_new[sess_id][:, k], label="State " + str(k + 1), lw=4,
                color=cols[k])
        #ax.plot(posterior_probs_true[sess_id][:, k], label="State " + str(k + 1), lw=2,
                #color=cols[k], linestyle='--')
    
    #plt.ylim((-0.01, 1.5))
    plt.legend(prop=dict(size=25))
    plt.yticks([0, 0.5, 1]) # had to remove  fontsize = 10 because mpl complained
    plt.tick_params(axis='both', which='major', labelsize=40) 
    plt.xlabel("trial #", fontsize = 50)
    plt.ylabel("p(state)", fontsize = 50)
    plt.title(f"States", fontsize=50)
    plt.savefig(f'states_{sess_id}_{experiment}.png')

def plot_choices(filt_inpts, filt_choices, sess_id):

    """Takes fitted glm hmm and axis object for plotting and plots choices on the third subplot"""

    
    
    ins = filt_inpts[i] # accessing 0th element in the outer list 
    ins = ins[0:, 0] # accessing entire column in the inputs 0th colunm, all rows
    cho = filt_choices[i]
    cho = cho[:, 0] # accessing entire column all rows
    mask = ins != 0 # create a mask to filter out 0s
    ins = ins[mask] # using mask filter out zeroes from ins
    cho = cho[mask] # filter out corresponding indexes from choices
    bool_inpts = (ins>0).astype(int) # convert inputs into bollean array first and then bools into numbers
    correct_choices = bool_inpts == cho # compare converted number to chouses
    plt.figure(figsize=(40,12))
    alpha = 0.6
    jitter = 0.05 * np.random.randn(len(cho))
    y_values_jittered = jitter+cho
    x_range = np.array(range(len(cho)))

    scatter_correct = plt.scatter(x_range[correct_choices], y_values_jittered[correct_choices], 
                                  label='correct', color='r', alpha=alpha, marker='v', s=200)
    scatter_wrong = plt.scatter(x_range[~correct_choices], y_values_jittered[~correct_choices],
                                 label='wrong', color='k',alpha=alpha, s=200)
    plt.yticks([0,1], ['L', 'R'])
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.legend(prop=dict(size=40))
    plt.xlabel("trial #", fontsize = 50)
    plt.ylabel("choice", fontsize = 50)
    plt.title(f"Choices", fontsize=50)
    plt.savefig(f'Choices{sess_id}_{experiment}.png')


def plot_all(hmm, filt_choices, filt_inputs, num_states, sess_id):

    """main plotting function. Plots states, choices, psytrack and glmhmm weights on different plots"""
    
    plot_states(hmm, filt_choices, filt_inputs, num_states, sess_id)
    plot_choices(filt_inputs, filt_choices, sess_id)
    

if __name__ == "__main__":
    args = get_args()
    choice_f: str = args.choices #holds path or file name of mouse choice data
    input_f: str = args.inputs #holds path or file name of mouse inputs data
    experiment: str = args.gname #holds experiment number to use as a graph header
         #load mouse inputs and choices

    inputs = np.load(input_f,allow_pickle = True) # load numpy file inputs
    choices = np.load(choice_f,allow_pickle = True) # load numpy files choices
    
    
    
    input_dim: int = inputs[0].shape[1]                                    # input dimensions
    num_states = 3        # number of discrete states
    TOL: int = 10**-4 # tolerance 
    N_iters: int = 1000 # number of iterations for the fitting model

    filt_inputs, filt_choices = filter_no_response_cho(inputs, choices, cho=2)
    obs_dim: int = filt_choices[0].shape[1]          # number of observed dimensions
    num_categories: int = len(np.unique(np.concatenate(filt_choices)))    # number of categories for output
    print(f'{np.shape(filt_choices)}')
    print(f'{np.shape(filt_inputs)=}')
    hmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")
    fit_glmhmm = fit_glm_hmm(hmm, filt_choices, filt_inputs, N_iters,TOL)
    #copy_cho = copy.deepcopy(filt_choices)


    #psy_track_data = convert_data_for_psytrack(filt_inputs, copy_cho)
    for i in range(len(filt_choices)):
        plot_all(hmm, filt_choices, filt_inputs, num_states, i)
        



