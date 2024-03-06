#!/usr/bin/env python

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import ssm
import psytrack as psy
import copy
import sim_psy_data
from ssm.util import find_permutation
import argparse
from collections import defaultdict
 


def get_args():
    """This grabs arguments for setting total trials"""
    parser = argparse.ArgumentParser(description="global variables to set input file names")
    parser.add_argument("-n", "--ntrials", help="starting number of trials per session", required=True, type=int)
    #parser.add_argument("-e", "--end", help="ending number of trials", required=True, type=int)
    #parser.add_argument("-i", "--increment", help="increments of trials", required=True, type=int)
    parser.add_argument("-se", "--session", help="total number of sessions", required=True, type=int)
    parser.add_argument("-o", "--outfile", help="output file name", required=True, type=str)
    return parser.parse_args()

def generate_inputs(num_trials_per_sess, num_sess):

    """Takes number of trials per session and number of sessions generates inputs from stimulus 
    values and choices returns inputs"""
    
    inpts = np.ones((num_sess, num_trials_per_sess, input_dim))  # initialize inpts array
    # set stimulus values negative value correct choice is left [0], positive value correct choice 
    # is right[1]
    stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25, 0.5, 1]
    inpts[:, :, 0] = np.random.choice(stim_vals, (num_sess, num_trials_per_sess))  # generate random sequence of stimuli
    inpts = list(inpts)  # convert inpts to correct format
    return inpts


def make_standard_hmm(gen_weights, gen_log_trans_mat ):

    """Takes a matrix of weights and a matrix of state transition probabilities, returns an HMM model
     initialized with those matrices """
    
    true_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")
    true_glmhmm.observations.params = gen_weights
    true_glmhmm.transitions.params = gen_log_trans_mat
    return true_glmhmm


def true_ll_model(true_glmhmm, num_trials_per_sess, num_sess, inpts):

    """"Takes number of trials per session and number of sessions returns log likelihood for the model
    initialized with standard weights and transition matrix""" 

    true_latents, true_choices = [], []
    for sess in range(num_sess):
        true_z, true_y = true_glmhmm.sample(num_trials_per_sess, input=inpts[sess])
        true_latents.append(true_z)
        true_choices.append(true_y)

    true_ll = true_glmhmm.log_probability(true_choices, inputs=inpts)
    return true_ll, true_latents, true_choices


def fit_glm_hmm(new_glmhmm, true_choice, inpts):

    """Takes newly initialized hmm glm model (without standard transition matrix and weights) and 
    returns a fitted glm hmm. Returns fitted glm-hmm"""

    fit_ll = new_glmhmm.fit(true_choice, inputs=inpts, method="em", num_iters=N_iters, tolerance=10**-4)

    return fit_ll


def convert_data_for_glmhmm(synth_psy_data):

    """Takes data dictionary that contains syntetic choice data generated by psytrack and converts it 
    into a format usable for glm-hmm. Returns choices array inthe format for fittiong glmhmm"""

    ps_data = copy.deepcopy(synth_psy_data['all_Y'])
    ps_data[0] = ps_data[0] -1 # subtract 1 from each element of psytrack data because it generates choices as 1 and 2
                                # and glm-hmm takes choices as 0 and 1
    ps_data[0] = ps_data[0][:, np.newaxis] # convert data into the glmhmm format numpy array
    #import ipdb; ipdb.set_trace()
    return ps_data

def get_glmhmm_dinmc_weights(new_glmhmm,inpts, true_choice):

    """Takes new glmhmm, generative glmhmm, inputs, choices generated by the generative glmhmm. 
    Calculates dinamic weights for each data point in glm-hmm model by multiplying 
    stimulus and bias values by posterior probability of the state in order to compare to psytrack.
    Returns a matrix of the weights: first column are results for stimulus and second are results for bias"""

    recovered_weights = new_glmhmm.observations.params

    
    posterior_probs_new = [new_glmhmm.expected_states(data=data, input=inpt)[0]
                for data, inpt
                in zip(true_choice, inpts)]
    
    # posterior_probs_true = [true_glmhmm.expected_states(data=data, input=inpt)[0]
    #             for data, inpt
    #             in zip(true_choice, inpts)]
    recovered_weights = np.squeeze(recovered_weights) # change dimensions of recovered weights to be (3,2)
    # perform matrix multiplication by multiplying posterior_probs_new[0] which is (11, 3) and recovered weights
    dinmc_weights_new = posterior_probs_new[0]@ recovered_weights 
    return dinmc_weights_new


def plot_dinamic_weights(ax, dinmc_weights_new, wMode, true_psy_we=None):
    
    """Takes    . Plots psytrack and dinamic hmmglm weights on the same plot"""
    #import ipdb; ipdb.set_trace()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    x_range = np.array(range(ntrials))
    #import ipdb; ipdb.set_trace()
    ax.plot(x_range, dinmc_weights_new[:, 0], c='tab:blue', label=f'glmhmm stimulus weights', linewidth=5)
    #ax.plot(x_range, dinmc_weights_new[:, 1], c='tab:orange', label=f'glmhmm_bias')
    #ax.plot(x_range, wMode[0], c='tab:green', label=f'psytrack_bias')
    ax.plot(x_range, wMode[1], c='tab:red', label=f'psytrack stimulus weigts', linewidth=5)
    if true_psy_we is not None:
        ax.plot(x_range, (true_psy_we['W'][:,0])*-1, c='k', label=f'true psytrack weights')
    ax.legend(prop=dict(size=25))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel("trial #", fontsize = 25)
    ax.set_ylabel("weight", fontsize = 25)
        


def convert_data_for_psytrack(inpts, choi_psy):

    """"Takes inputs and choices generated by glmhmm. Converts iinput and choice data into appropriate
      formats for fitting psytrack, returns a dictionary with inputs and choices"""
    
    psy_data_dict = defaultdict(lambda: 'Not present')
    psy_data_dict['y'] = np.squeeze(choi_psy[0]) # add choices data with removing the lowest dimension in choice data
    ins = inpts[0]
    # in the original notebook they add inputs in the following way: the inputs is a 2D array at the lowest
    # level. First column is a column of inputs which are stimulus values, the second column is a shifted 
    # first column in a way: first column: (0, 1, 2, 3) second would be (0, 0, 1, 2)
    psy_data_dict['inputs'] = {'inpt1': np.column_stack((ins[0:, 0], np.roll(ins[0:, 0], 1)))} # add input data into a dictionary of inputs
    psy_data_dict['inputs']['inpt1'][0,1] = psy_data_dict['inputs']['inpt1'][0,0] # update the first first value in the second column
    return(psy_data_dict)


def generate_data_psytrack(ntrials, stim_list):
    """"Takes   . Generates choice data using psytrack model"""

    seed = 31
    num_weights = 1
    #num_trials = 5000
    hyper = {'sigma'   : 2**np.array([-4.0]), #np.array([-4.0,-5.0,-6.0,-7.0])
         'sigInit' : 2**np.array([ 0.0])} # p.array([ 0.0, 0.0, 0.0, 0.0])
    simData = sim_psy_data.generateSimData(K=num_weights, N=ntrials, hyper=hyper, stim_list=stim_list,
                          boundary=6.0, iterations=1, seed=seed, savePath=None)
    return simData

def fit_psytrack(psy_track_data, inp_name: str):

    """Takes psytrack data dictionary. Fits psytrack with choice and input data. Returns:
    hyp: a dictionary of the optimized hyperparameters
    evd: the approximate log-evidence of the optimized model
    wMode: the weight trajectories of the optimized model
    hess_info a dictionary of sparse terms that relate to the Hessian of the optimal model
    """
    
    
    # generate a dictionary of weights for the model
    weights = {'bias': 1,
               inp_name: 1} # key is the one of the inputs key from the data dictionary, value is how
                        # many columns if that inputs matrix should be used

    K = np.sum([weights[i] for i in weights.keys()]) # the total number of weights K in the model

    # generate a dictionary of hyperparameters
    hyper= {'sigInit': 2**4.,      # Set to a single, large value for all weights. Will not be optimized further.
        'sigma': [2**-4.]*K,   # Each weight will have it's own sigma optimized, but all are initialized the same
        'sigDay': None}        # Indicates that session boundaries will be ignored in the optimization
    
    # set parameters to optimize over. If optList is empty, there will be no optimization
    optList = ['sigma'] # optimize over sigma parameter 
    # fit psytrack
    hyp, evd, wMode, hess_info = psy.hyperOpt(psy_track_data, hyper, weights, optList)
    print('finished fitting psytrack')
    return hyp, evd, wMode, hess_info

def plot_choices(ax, inpts, true_choice):

    """Takes fitted glm hmm and axis object for plotting and plots choices on the third subplot"""

    ins = inpts[0] # accessing 0th element in the outer list 
    ins = ins[0:, 0] # accessing entire column in the inputs 0th colunm, all rows
    cho = true_choice[0]
    cho = cho[:, 0] # accessing entire column all rows
    mask = ins != 0 # create a mask to filter out 0s
    ins = ins[mask] # using mask filter out zeroes from ins
    cho = cho[mask] # filter out corresponding indexes from choices
    bool_inpts = (ins>0).astype(int) # convert inputs into bollean array first and then bools into numbers
    correct_choices = bool_inpts == cho # compare converted number to chouses

    alpha = 0.6
    jitter = 0.1 * np.random.randn(len(cho))
    y_values_jittered = jitter+cho
    x_range = np.array(range(len(cho)))
    scatter_correct = ax.scatter(x_range[correct_choices], y_values_jittered[correct_choices], 
                                  label='correct', color='r', alpha=alpha, marker='v', s=50)
    scatter_wrong = ax.scatter(x_range[~correct_choices], y_values_jittered[~correct_choices],
                                 label='wrong', color='k',alpha=alpha, s=50)
    ax.set_yticks([0,1], ['L', 'R'])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(prop=dict(size=25))
    ax.set_xlabel("trial #", fontsize = 25)
    ax.set_ylabel("choice", fontsize = 25)
    ax.set_title(f"Choices (Set {ntrials})", fontsize=25)


def plot_states(new_glmhmm,  true_glmhmm, ax, true_choice, inpts, true_latents):

    """Takes fitted glm hmm and an axis object for plotting and plots states on the second subplot"""

    cols = ['#ff7f00', '#4daf4a', '#377eb8']
    posterior_probs_new = [new_glmhmm.expected_states(data=data, input=inpt)[0]
                for data, inpt
                in zip(true_choice, inpts)] # plot states predicted by the model not initialized with statndard weights and matrices
    #posterior_probs_true = [true_glmhmm.expected_states(data=data, input=inpt)[0]
                #for data, inpt
                #in zip(true_choice, inpts)] # plot states predicted by the model initialized with standard weights and matrices
    
    # plot true states of the model when it generated the choice data
    x_range = np.array(range(len(true_latents[0])))
    for i in range(len(cols)):
        mask = (true_latents[0] == i)
        y_values = np.ones(len(true_latents[0][mask]))
        ax.scatter(x_range[mask], y_values*1.25, c=cols[i], label=f'state {i+1}')


    sess_id = 0 #session id; can choose any index between 0 and num_sess-1
    for k in range(num_states):
        ax.plot(posterior_probs_new[sess_id][:, k], label="State " + str(k + 1), lw=4,
                color=cols[k])
        #ax.plot(posterior_probs_true[sess_id][:, k], label="State " + str(k + 1), lw=2,
                #color=cols[k], linestyle='--')
    ax.set_ylim((-0.01, 1.5))
    ax.legend(prop=dict(size=25))
    ax.set_yticks([0, 0.5, 1]) # had to remove  fontsize = 10 because mpl complained
    ax.tick_params(axis='both', which='major', labelsize=20) 
    ax.set_xlabel("trial #", fontsize = 25)
    ax.set_ylabel("p(state)", fontsize = 25)
    ax.set_title(f"States (Set {ntrials})", fontsize=25)


def plot_weights(new_glmhmm, ax, gen_weights):

    """Takes fitted glm hmm and axis object for plotting and plots standard and recovered weights 
    on the first of three subplots"""

    #fig = plt.figure(figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    cols = ['#ff7f00', '#4daf4a', '#377eb8'] # '#ff7f00' dark orange, #4daf4a green, '#377eb8' blue
    recovered_weights = new_glmhmm.observations.params
    
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
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(prop=dict(size=25))
    ax.set_title(f"Weight recovery (Set {ntrials})", fontsize=25)


def psy_with_glmhmm_cho():
    """Takes   . Uses choices generated by glmhmm to fit the model and plot results"""

    gen_weights = np.array([[[6, 1]], [[2, -3]], [[2, 3]]])
    gen_log_trans_mat = np.log(np.array([[[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.03, 0.03, 0.94]]]))
    true_glmhmm = make_standard_hmm(gen_weights, gen_log_trans_mat)

   
    # initialize new glm_hmm
    new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                            observation_kwargs=dict(C=num_categories), transitions="standard")
    inpts = generate_inputs(ntrials, num_sess)
    true_ll, true_latents, true_choice = true_ll_model(true_glmhmm, ntrials, num_sess, inpts) # geterate choices using glm-hmm
    fit_ll = fit_glm_hmm(new_glmhmm, true_choice, inpts) # fit newly created glm-hmm

    perm = find_permutation(true_latents[0], new_glmhmm.most_likely_states(true_choice[0], input=inpts[0]),
                                 K1=num_states, K2=num_states)
    
    new_glmhmm.permute(perm) # find permutations for the glmhmm

    choi_psy = copy.deepcopy(true_choice)
    psy_track_data = convert_data_for_psytrack(inpts, choi_psy) # convert data generated with glmhmm into psytrack format
    hyp, evd, wMode, hess_info = fit_psytrack(psy_track_data, 'inpt1') # fit psytrack
    dinmc_weights_new = get_glmhmm_dinmc_weights(new_glmhmm, true_glmhmm, inpts, true_choice) # get dynamic weights for glmhmm


    fig, axes = plt.subplots(nrows=4, figsize=(400, 30), dpi=80, facecolor='w', edgecolor='k')

    for i in range(4):
        # if i == 0:
        #     plot_weights(new_glmhmm, axes[i], gen_weights) # plot weights
        if i == 1:
            plot_states(new_glmhmm,  true_glmhmm, axes[i], true_choice, inpts, true_latents) # plot states
        if i == 2:
            plot_choices(axes[i], inpts, true_choice) # plot choices
        if i == 3:
            plot_dinamic_weights(axes[i], dinmc_weights_new, (wMode)*-1)
    plt.tight_layout()
    plt.savefig(f'glmhmm_psytrack_{out}_session_{ntrials}_.png')


def glmhmm_with_psy_cho(synth_psy_data):

    """Takes synth_psy_data: synthetic data generated by psytrack in a form of dict where 'W" is an array 
    of psycometric weights, 'X' is array of inputs generated from normal random, and "all_Y" is an 
    array of choices encoded as 1 and 2s. glmhmm_data is choice data extracted from "all_Y" and converted into format 
    suitable for glm-hmm.
    Uses choice data generated by psytrack to fit the glmhmm and psytrack models and plot
    dynamic weights"""

    #import ipdb; ipdb.set_trace()
    glmhmm_data_choices = convert_data_for_glmhmm(synth_psy_data) # convert data generated by psytrack into glm-hmm format
    new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                            observation_kwargs=dict(C=num_categories), transitions="standard")
    
    #convert simulated psytrack data into the data (dict) suitable for fitting psytrack
    psy_data_dict = defaultdict(lambda: 'Not present')
    psy_data_dict['y'] = synth_psy_data['all_Y'][0]
    psy_data_dict['inputs'] = {'inpt1': synth_psy_data['X']} # add input data into a dictionary of inputs
    
    hyp, evd, wMode, hess_info = fit_psytrack(psy_data_dict, 'inpt1') # fit psytrack

    # convert psytrack inputs and biases into glm-hmm inputs
    
    stimulus = synth_psy_data['X'][:,0]
    bias = wMode[0]
    # fit glmhmm
    glm_hmm_inpts = [np.concatenate((stimulus[:, np.newaxis], bias[:, np.newaxis]), axis=1)]
    fit_ll = fit_glm_hmm(new_glmhmm, glmhmm_data_choices, glm_hmm_inpts)
    glmhmm_dyn_weights = get_glmhmm_dinmc_weights(new_glmhmm, glm_hmm_inpts, glmhmm_data_choices)
    fig, axes = plt.subplots(nrows=1, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    #import ipdb; ipdb.set_trace()

    plot_dinamic_weights(axes, glmhmm_dyn_weights, (wMode)*-1, synth_psy_data)
    plt.tight_layout()
    plt.savefig(f'psdata_{out}_session_{ntrials}_.png')


if __name__ == "__main__":
    args = get_args()
    ntrials = args.ntrials #  number of trials
    #end = args.end  # Use the 'end' argument to specify the ending number of trials
    #inc = args.increment # increment number of trials
    num_sess = args.session # number of sessions
    out = args.outfile # name for plot figure


    # Set the parameters of the GLM-HMM
    num_states = 3        # number of discrete states
    obs_dim = 1           # number of observed dimensions
    num_categories = 2    # number of categories for output
    input_dim = 2         # input dimensions
    N_iters = 1000        # number of fit iterations 
    npr.seed(0)
    # set the weights and state transition probability matrix
    
    stim_list = [-1, -0.5, -0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25, 0.5, 1]
    # initialize a standard glmhmm with starting weights and transition matrix

    
    
    
    synth_psy_data = generate_data_psytrack(ntrials, stim_list) # generate data using psytrack
    
    glmhmm_with_psy_cho(synth_psy_data)
    #psy_with_glmhmm_cho()
    

    
    


    

