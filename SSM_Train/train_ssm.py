#!/usr/bin/env python

# the purpose of this script is to fit glm-hmm to mouse choice data
# and output the posterior probability which is glm-hmm state probability 
# into a csv file for subsequent protting or other data manipulations
# use this line for debugging: import ipdb; ipdb.set_trace()


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
    parser.add_argument("-rt", "--react_times", help="reaction times file name", required=True, type=str)
    parser.add_argument("-g", "--gname", help="graph name", required=True, type=str)
    return parser.parse_args()


def fit_glm_hmm(hmm, filt_choices, filt_inpts, N_iters,TOL):

    """Takes newly initialized hmm glm model (without standard transition matrix and weights) and 
    returns a fitted glm hmm. Returns fitted glm-hmm"""

    fit_glmhmm = hmm.fit(np.concatenate(filt_choices), inputs=np.concatenate(filt_inpts), method="em",
                          num_iters=N_iters, tolerance=TOL)
    
    return fit_glmhmm

def get_state_probs(hmm, choices, inputs):

    """Takes trained model, mouse choices, and inputs. Returns all state probabilities as a list of numpy array
    each array in the list represents a session. Each trial in a session contains a triplet of state probabilities
    representing a probability of each state"""

    posterior_probs_new = [hmm.expected_states(data=data, input=inpt)[0]
                for data, inpt
                in zip(choices, inputs)]
    return posterior_probs_new


def plot_states(ax, hmm, filt_choices, filt_inputs, num_states, sess_id = None):

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
    for k in range(num_states):
        ax.plot(posterior_probs_new[sess_id][:, k], label="State " + str(k + 1), lw=4,
                color=cols[k])
        #ax.plot(posterior_probs_true[sess_id][:, k], label="State " + str(k + 1), lw=2,
                #color=cols[k], linestyle='--')
    
    #plt.ylim((-0.01, 1.5))
    ax.legend(prop=dict(size=25))
    ax.set_yticks([0, 0.5, 1]) # had to remove  fontsize = 10 because mpl complained
    ax.tick_params(axis='both', which='major', labelsize=40) 
    ax.set_xlabel("trial #", fontsize = 50)
    ax.set_ylabel("p(state)", fontsize = 50)
    ax.set_title(f"States", fontsize=50)


def plot_reaction_times(ax, react_times, sess_id):

    """Takes reaction times and axes and plots them on it's own subplot"""

    x_range = np.arange(len(react_times[sess_id]))
    ax.plot(x_range, np.squeeze(react_times[sess_id]), color = 'k')
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.set_xlabel("trial #", fontsize = 50)
    ax.set_ylabel("reaction time", fontsize = 50)
    ax.set_title(f"Reaction times", fontsize=50)




def plot_all(trans_list, sess_id):

    """Takes a list of indexes where state transitions happen and plot reaction times on a histogram"""

    fig = plt.figure(figsize=(100, 50), dpi=80, facecolor='w', edgecolor='k')
    gs=GridSpec(3,len(trans_list))
    c: int = 0 # counter to count the number of histograms
    for i in range(len(trans_list+1)):
        if i == 0:
            ax = fig.add_subplot(gs[0,:])
            plot_reaction_times(ax, react_times, sess_id)
        if i == 1:
            ax = fig.add_subplot(gs[1,:])
            ax = plot_states(ax, hmm, choices, inputs, num_states, sess_id)
        if i > 2:
            ax = fig.add_subplot(gs[2,c])
            plot_peristimulus_hist(ax, trans_list[i])
            c+=1
    plt.savefig(f'glmhmm_rt_fit_{experiment}_session__.png')


def get_rts(trans_list, sess_id, rts_list_before, rts_list_after, react_times, wind_sze):

    """Takes a list of transitions that contains indexes of points when state transition happened.
    Accesses reaction time numpy array ussing session number and saves corresponding reaction times into a reaction times list"""

    
    for tr in trans_list:
        if tr >= wind_sze:
            befors = react_times[sess_id][tr-wind_sze:tr]
            rts_list_before = rts_list_before + befors.tolist()
            
        else:
            befors = react_times[sess_id][0:tr]
            rts_list_before = rts_list_before + befors.tolist()
            
        if len(react_times[sess_id]) - tr > wind_sze:
            afters = react_times[sess_id][tr+1:tr+wind_sze+1]
            rts_list_after = rts_list_after + afters.tolist()
            
        else:
            afters = react_times[sess_id][tr+1:]
            rts_list_after = rts_list_after + afters.tolist()
        
    return rts_list_before, rts_list_after


def parse_probs(state_probs, react_times, wind_sze):

    """Takes state probabilities. Iterates over state probabilities by session and triggers plotting function if
    state probability drops lower than 80% or rases higher than 80%"""

    drop_trans = []
    raise_trans = []
    rts_before_drop = []
    rts_after_drop = []
    rts_before_raise = []
    rts_after_raise = []
    for sess_id in range(len(state_probs)): 
        max_probs = np.max(state_probs[sess_id], axis=1)
        for i in range(len(max_probs)):
            if (i != 0) or (i != (len(max_probs)-1)):
                if max_probs[i] <= 0.80 and max_probs[i-1] >= 0.80:
                    drop_trans.append(i)
                if max_probs[i] >= 0.80 and max_probs[i-1] <= 0.8:
                    raise_trans.append(i)
        rts_before_drop, rts_after_drop = get_rts(drop_trans, sess_id,rts_before_drop, rts_after_drop, react_times, wind_sze)
        #print(f'drop {len(rts_before_drop)=} {len(rts_after_drop)}')
        rts_before_raise, rts_after_raise = get_rts(raise_trans, sess_id, rts_before_raise, rts_after_raise, react_times, wind_sze)
        #print(f'raise {len(rts_before_raise)=} {len(rts_after_raise)}')
        drop_trans = []
        raise_trans = []


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(80, 40), dpi=80, facecolor='w', edgecolor='k')  
    plot_peristimulus_hist(axes, rts_before_drop, rts_after_drop, col='tab:purple')
    plt.tight_layout()
    plt.savefig(f'peristim_hist_drop_{experiment}_.png')
    plt.close(fig) # close previous figure otherwise computer runs out of memory

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(80, 40), dpi=80, facecolor='w', edgecolor='k')  
    plot_peristimulus_hist(axes, rts_before_raise, rts_after_raise, col='tab:green')
    plt.tight_layout()
    plt.savefig(f'peristim_hist_rise_{experiment}_.png')
    plt.close(fig) # close previous figure otherwise computer runs out of memory


def plot_peristimulus_hist(axes, rt_before, rt_after, col):
    
    """Takes reaction time array and plots a histogram of reaction times before and after state transition point"""

    
    axes[0].hist(rt_before, bins=int(math.sqrt(len(rt_before))), edgecolor='k', color=col)
    axes[0].tick_params(axis='both', which='major', labelsize=40)
    axes[0].set_xlabel("reaxtion times", fontsize = 50)
    axes[0].set_ylabel("count", fontsize = 50)
    axes[0].set_title(f"Reaction times before transition", fontsize=50)
    filt_rt_before = [x for x in rt_before if not math.isnan(x)] # filter out nans
    avrg_before = sum(filt_rt_before)/len(filt_rt_before)
    axes[0].axvline(avrg_before, color='k', linestyle='dashed', linewidth=10)
    min_ylim, max_ylim = axes[0].set_ylim()
    plt.text(avrg_before*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(avrg_before), fontsize=50)

    axes[1].hist(rt_after, bins=int(math.sqrt(len(rt_after))), edgecolor='k', color=col)
    axes[1].tick_params(axis='both', which='major', labelsize=40)
    axes[1].set_xlabel("reaxtion times", fontsize = 50)
    axes[1].set_ylabel("count", fontsize = 50)
    axes[1].set_title(f"Reaction times after transition", fontsize=50)
    filt_rt_after = [x for x in rt_after if not math.isnan(x)] # filter out nans
    avrg_after = sum(filt_rt_after)/len(filt_rt_after)
    axes[1].axvline(avrg_after, color='k', linestyle='dashed', linewidth=10)
    min_ylim, max_ylim = axes[1].set_ylim()
    plt.text(avrg_after*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(avrg_after), fontsize=50)

def parse_probs_by_state(sess_state_prob):

    """Takes . Parses state probabilities by finding the max out of three probabilities for each data point
    and saving it and its corresponding state in a tuple"""

    max_probs_and_inds = []
    for row in sess_state_prob:
        max_prob = np.max(row) # max probability out of three
        col_ind = np.argmax(row) # index of the max probability
        max_probs_and_inds.append((max_prob, col_ind))

    return max_probs_and_inds
    



if __name__ == "__main__":
    args = get_args()
    choice_f: str = args.choices #holds path or file name of mouse choice data
    input_f: str = args.inputs #holds path or file name of mouse inputs data
    react_times_f = args.react_times #holds path for reaction times
    experiment: str = args.gname #holds experiment number to use as a graph header
         #load mouse inputs and choices
    npr.seed(42)

    inputs = np.load(input_f,allow_pickle = True) # load numpy file inputs
    choices = np.load(choice_f,allow_pickle = True) # load numpy files choices
    react_times = np.load(react_times_f, allow_pickle = True) # load numpy file reaction times
    input_dim: int = inputs[0].shape[1]  # input dimensions
    num_states = 3        # number of discrete states
    TOL: float = 10**-4 # tolerance 
    N_iters: int = 1000 # number of iterations for the fitting model

    obs_dim: int = choices[0].shape[1]          # number of observed dimensions
    num_categories: int = len(np.unique(np.concatenate(choices)))    # number of categories for output

    hmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")
    fit_glmhmm = fit_glm_hmm(hmm, choices, inputs, N_iters,TOL)
    state_probs = get_state_probs(hmm, choices, inputs)

    wind_sze = 5
    #parse_probs(state_probs, react_times, wind_sze)

    for sess_id in range(len(state_probs)):
        max_probs_inds = parse_probs_by_state(state_probs[sess_id])
        break
    import ipdb; ipdb.set_trace()
    

