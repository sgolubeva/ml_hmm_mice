#!/usr/bin/env python
#The purpose of this script is to compare glm-hmm performance on small to large amount of data points 
# To acomplish this, we create a standard glm-hmm model initialized with preset transition matrix and 
# weights. Then we generate syntetic data for different numbers of sessions and trials and initialize
# and fit a new glm-hmm model. Then we compare weights and biases from the model initialized with
# known parameters vs weights and biases recovered from each new model and plot them. Additionally,
# we plot states determined by the model and choices to investigate how exactly model classifies states
# based on choices. 

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import ssm
from ssm.util import find_permutation
import argparse
import IPython 

npr.seed(0)

def get_args():
    """This grabs arguments for setting total trials"""
    parser = argparse.ArgumentParser(description="global variables to set input file names")
    parser.add_argument("-s", "--start", help="starting number of trials per session", required=True, type=int)
    parser.add_argument("-e", "--end", help="ending number of trials", required=True, type=int)
    parser.add_argument("-i", "--increment", help="increments of trials", required=True, type=int)
    parser.add_argument("-se", "--session", help="total number of sessions", required=True, type=int)
    parser.add_argument("-o", "--outfile", help="output file name", required=True, type=str)
    return parser.parse_args()


def make_standard_hmm(gen_weights, gen_log_trans_mat ):

    """Takes a matrix of weights and a matrix of state transition probabilities, returns an HMM model
     initialized with those matrices """
    true_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")
    true_glmhmm.observations.params = gen_weights
    true_glmhmm.transitions.params = gen_log_trans_mat
    return true_glmhmm

def generate_inputs(num_trials_per_sess):

    """Takes number of trials per session generates inputs from stimulus values and choices returns
    inputs"""


    print(f'generate ipts{num_trials_per_sess=}')
    inpts = np.ones((num_sess, num_trials_per_sess, input_dim))  # initialize inpts array
    stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]
    inpts[:, :, 0] = np.random.choice(stim_vals, (num_sess, num_trials_per_sess))  # generate random sequence of stimuli
    inpts = list(inpts)  # convert inpts to correct format
    return inpts

def true_ll_model(num_trials_per_sess, num_sess):

    """"Takes number of trials per session and number of sessions returns log likelihood for the model
    initialized with standard weights and transition matrix""" 
    true_latents, true_choices = [], []
    for sess in range(num_sess):
        true_z, true_y = true_glmhmm.sample(num_trials_per_sess, input=inpts[sess])
        true_latents.append(true_z)
        true_choices.append(true_y)

    true_ll = true_glmhmm.log_probability(true_choices, inputs=inpts)
    return true_ll, true_latents, true_choices

def fit_glm_hmm(new_glmhmm):

    """Takes newly initialized hmm glm model (without standard transition matrix and weights) and 
    returns a fitted glm hmm"""

    fit_ll = new_glmhmm.fit(true_choices, inputs=inpts, method="em", num_iters=N_iters, tolerance=10**-4)

    return fit_ll


def plot_weights(new_glmhmm, ax):

    """Takes fitted glm hmm and axis object for plotting and plots standard and recovered weights 
    on the first of three subplots"""

    #fig = plt.figure(figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    cols = ['#ff7f00', '#4daf4a', '#377eb8']
    recovered_weights = new_glmhmm.observations.params
    #print(recovered_weights)
    for k in range(num_states):
        if k == 0:
            ax.plot(range(input_dim), gen_weights[k][0], marker='o',
                    color=cols[k], linestyle='-',
                    lw=1.5, label="generative")
            ax.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                        lw=1.5,  label="recovered", linestyle='--')
        else:
            ax.plot(range(input_dim), gen_weights[k][0], marker='o',
                    color=cols[k], linestyle='-',
                    lw=1.5, label="")
            ax.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                        lw=1.5, label="", linestyle='--')
    #ax.set_yticks(fontsize=10)
    ax.set_ylabel("GLM weight", fontsize=25)
    ax.set_xlabel("covariate", fontsize=25)
    ax.set_xticks([0, 1], ['stimulus', 'bias'], fontsize=25, rotation=25)
    ax.axhline(y=0, color="k", alpha=0.5, ls="--")
    ax.legend()
    ax.set_title(f"Weight recovery (Set {j})", fontsize=25)
    #plt.savefig(f'{out}_{j}.png')
    #return fig

def plot_states(new_glmhmm, ax):

    """Takes fitted glm hmm and an axis object for plotting and plots states on the second subplot"""

    cols = ['#ff7f00', '#4daf4a', '#377eb8']
    posterior_probs = [new_glmhmm.expected_states(data=data, input=inpt)[0]
                for data, inpt
                in zip(true_choices, inpts)]
    #fig = plt.figure(figsize=(15, 2.5), dpi=80, facecolor='w', edgecolor='k')
    sess_id = 0 #session id; can choose any index between 0 and num_sess-1
    for k in range(num_states):
        ax.plot(posterior_probs[sess_id][:, k], label="State " + str(k + 1), lw=2,
                color=cols[k])
    ax.set_ylim((-0.01, 1.01))
    ax.set_yticks([0, 0.5, 1]) # had to remove  fontsize = 10 because mpl complained 
    ax.set_xlabel("trial #", fontsize = 25)
    ax.set_ylabel("p(state)", fontsize = 25)
    ax.set_title(f"States (Set {j})", fontsize=25)
    #plt.savefig(f'states_fig{out}session{j}.png')
    #return fig

def plot_choices(n_sess, ax):

    """Takes fitted glm hmm and axis object for plotting and plots choices on the third subplot"""

    ins = inpts[0] # accessing 0th element in the outer list 
    ins = ins[0:, 0] # accessing entire column in the inputs 0th colunm, all rows
    cho = true_choices[0]
    cho = cho[:, 0] # accessing entire column all rows
    mask = ins != 0 # create a mask to filter out 0s
    ins = ins[mask] # using mask filter out zeroes from ins
    cho = cho[mask] # filter out corresponding indexes from choices
    bool_inpts = (ins>0).astype(int) # convert inputs into bollean array first and then bools into numbers
    correct_choices = bool_inpts == cho # compare converted number to chouses

    colors = ['black', 'red']
    alpha = 0.6
    jitter = 0.1 * np.random.randn(len(cho))
    y_values_jittered = jitter+cho
    #fig = plt.figure(figsize=(15, 2.5), dpi=80, facecolor='w', edgecolor='k')
    #ax = plt.axes()
    x_range = np.array(range(len(cho)))
    scatter_correct = ax.scatter(x_range[correct_choices], y_values_jittered[correct_choices], 
                                  label='correct', color='r', alpha=alpha)
    scatter_wrong = ax.scatter(x_range[~correct_choices], y_values_jittered[~correct_choices],
                                 label='wrong', color='k',alpha=alpha)
    ax.set_yticks([0,1], ['L', 'R'])
    ax.legend()
    ax.set_xlabel("trial #", fontsize = 25)
    ax.set_ylabel("choice", fontsize = 25)
    ax.set_title(f"Choices (Set {j})", fontsize=25)
    #plt.savefig(f'choices_fig{out}session{j}.png')
    #return fig
    


if __name__ == "__main__":
    args = get_args()
    start = args.start # starting number of trials
    end = args.end  # Use the 'end' argument to specify the ending number of trials
    inc = args.increment # increment number of trials
    num_sess = args.session # number of sessions
    out = args.outfile # name for plot figure

    # Set the parameters of the GLM-HMM
    num_states = 3        # number of discrete states
    obs_dim = 1           # number of observed dimensions
    num_categories = 2    # number of categories for output
    input_dim = 2         # input dimensions
    N_iters = 1000          # number of fit iterations 
    # set stimulus values negative value correct choice is left [0], positive value correct choice 
    # is right[1]
    

    # set weights and transition matrix numbers for the HMM model initiated with specific starting
    #  parameters
    gen_weights = np.array([[[6, 1]], [[2, -3]], [[2, 3]]])
    gen_log_trans_mat = np.log(np.array([[[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.03, 0.03, 0.94]]]))
    # initialize an glmhmm with specific parameters
    true_glmhmm = make_standard_hmm(gen_weights, gen_log_trans_mat)

    for j in range(start, end + 1, inc):
        inpts=generate_inputs(j)
        true_ll, true_latents, true_choices = true_ll_model(j, num_sess)
      
        # inialize new hmm without standard input values
        new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                            observation_kwargs=dict(C=num_categories), transitions="standard")
        
        #new_glmhmm.permute(find_permutation(true_latents[0], new_glmhmm.most_likely_states(true_choices[0], input=inpts[0])))    
        fit_ll = fit_glm_hmm(new_glmhmm) # fit newly created glm-hmm
        # create figure an axes with 3 subplots for plotting
        fig, axes = plt.subplots(nrows=3, figsize=(30, 30), dpi=80, facecolor='w', edgecolor='k')
        for i in range(3):
            if i == 0:
                plot_weights(new_glmhmm, axes[i]) # plot weights
            if i == 1:
                plot_states(new_glmhmm, axes[i]) # plot states
            if i == 2:
                plot_choices(j, axes[i]) # plot choices
        plt.tight_layout()
        plt.savefig(f'wsc_fig{out}session{j}.png')
        
