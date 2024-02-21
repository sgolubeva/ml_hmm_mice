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
import statistics
from ssm.util import find_permutation
import argparse
from collections import defaultdict
import copy
import math
from matplotlib.gridspec import GridSpec
from collections import defaultdict



def get_args():

    """This grabs arguments for setting total trials"""

    parser = argparse.ArgumentParser(description="global variables to set input file names")
    parser.add_argument("-ch", "--choices", help="choices file name", required=True, type=str)
    #parser.add_argument("-ip", "--inputs", help="input file name", required=True, type=str)
    parser.add_argument("-st", "--stimilus", help="stimulus numpy file naame", required=True, type=str)
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
            plot_reaction_times(ax, react_times_exp, sess_id)
        if i == 1:
            ax = fig.add_subplot(gs[1,:])
            ax = plot_states(ax, hmm, choices, inputs, num_states, sess_id)
        if i > 2:
            ax = fig.add_subplot(gs[2,c])
            plot_peristimulus_hist(ax, trans_list[i])
            c+=1
    plt.savefig(f'glmhmm_rt_fit_{experiment}_session__.png')


def get_rts(trans_list, sess_id, rts_list_before, rts_list_after, react_times_exp, wind_sze):

    """Takes a list of transitions that contains indexes of points when state transition happened.
    Accesses reaction time numpy array ussing session number and saves corresponding reaction times into a reaction times list"""

    
    for tr in trans_list:
        if tr >= wind_sze:
            befors = react_times_exp[sess_id][tr-wind_sze:tr]
            rts_list_before = rts_list_before + befors.tolist()
            
        else:
            befors = react_times_exp[sess_id][0:tr]
            rts_list_before = rts_list_before + befors.tolist()
            
        if len(react_times_exp[sess_id]) - tr > wind_sze:
            afters = react_times_exp[sess_id][tr+1:tr+wind_sze+1]
            rts_list_after = rts_list_after + afters.tolist()
            
        else:
            afters = react_times_exp[sess_id][tr+1:]
            rts_list_after = rts_list_after + afters.tolist()
        
    return rts_list_before, rts_list_after


def parse_probs(state_probs, react_times_exp, wind_sze):

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
        rts_before_drop, rts_after_drop = get_rts(drop_trans, sess_id,rts_before_drop, rts_after_drop, react_times_exp, wind_sze)
        
        rts_before_raise, rts_after_raise = get_rts(raise_trans, sess_id, rts_before_raise, rts_after_raise, react_times_exp, wind_sze)
        
        drop_trans = []
        raise_trans = []


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(80, 40), dpi=80, facecolor='w', edgecolor='k')  
    plot_peristimulus_hist(axes, rts_before_drop, rts_after_drop, col='tab:purple')
    plt.tight_layout()
    plt.savefig(f'peristim_hists/{experiment}/peristim_hist_drop_{experiment}_.png')
    plt.close(fig) # close previous figure otherwise computer runs out of memory

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(80, 40), dpi=80, facecolor='w', edgecolor='k')  
    plot_peristimulus_hist(axes, rts_before_raise, rts_after_raise, col='tab:green')
    plt.tight_layout()
    plt.savefig(f'peristim_hists/{experiment}/peristim_hist_rise_{experiment}_.png')
    plt.close(fig) # close previous figure otherwise computer runs out of memory


def plot_peristimulus_hist(axes, rt_before, rt_after, col):
    
    """Takes reaction time array and plots a histogram of reaction times before and after state transition point"""

    
    axes[0].hist(rt_before, bins=int(math.sqrt(len(rt_before))), edgecolor='k', color=col)
    axes[0].tick_params(axis='both', which='major', labelsize=40)
    axes[0].set_xlabel("reaction times", fontsize = 50)
    axes[0].set_ylabel("count", fontsize = 50)
    axes[0].set_title(f"Reaction times before transition", fontsize=50)
    filt_rt_before = [x for x in rt_before if not math.isnan(x)] # filter out nans
    avrg_before = sum(filt_rt_before)/len(filt_rt_before)
    axes[0].axvline(avrg_before, color='k', linestyle='dashed', linewidth=10)
    min_ylim, max_ylim = axes[0].set_ylim()
    plt.text(avrg_before*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(avrg_before), fontsize=50)

    axes[1].hist(rt_after, bins=int(math.sqrt(len(rt_after))), edgecolor='k', color=col)
    axes[1].tick_params(axis='both', which='major', labelsize=40)
    axes[1].set_xlabel("reaction times", fontsize = 50)
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
    
def find_state_change_inds(max_probs, drop_trans, raise_trans):

    """Takes a list of tuples of max probabilities for each data point and it's corresponding state
    finds indexes of state switching point"""

    for i in range(len(max_probs)):
        if (i != 0) or (i != (len(max_probs)-1)):
            if max_probs[i][0] <= 0.80 and max_probs[i-1][0] >= 0.80:
                drop_trans.append((i, max_probs[i][1]))
            if max_probs[i][0] >= 0.80 and max_probs[i-1][0] <= 0.8:
                raise_trans.append((i, max_probs[i][1]))
    return drop_trans, raise_trans                              

def combine_probs_by_trans(drop_trans, raise_trans, states_dict, sess_id):

    """Takes: drop_trans - the list of indexes where the probability of a state drops lower 80%, and corresponding state number
    raise_trans indexes where the state probability gets higher than 80% and corresponding state. Adds probability indexes into a dict
    under a tuple key (previous state, new state)"""

    
    for i in range(len(drop_trans)):
        key = (raise_trans[i][1])
        states_dict[key].append((drop_trans[i][0], raise_trans[i][0], sess_id))
    

def get_rt_values(react_times_exp, states_dict, window):

    """Takes dictionary with state transition probability indexes and uses them to get the reaction times for 
    before and after the  state change"""

    position_before = defaultdict(list)
    position_after = defaultdict(list)
    for key in states_dict:
        
        for item in states_dict[key]:
            #import ipdb; ipdb.set_trace() 
            tr = item[0]
            sess = item[2]
            if tr >= window:
                befors = react_times_exp[sess][tr-window:tr]
                
                for i in range(len(befors)):
                    position_before[i].append(befors[i])

            else:
                befors = react_times_exp[sess][0:tr]
                
                for i in range(len(befors)):
                    position_before[i].append(befors[i])

            if len(react_times_exp[sess]) - tr > window:
                afters = react_times_exp[sess][tr+1:tr+window+1]
                for i in range(len(afters)):
                    position_after[i].append(afters[i])

            else:
                afters = react_times_exp[sess][tr+1:]
                for i in range(len(afters)):
                    position_after[i].append(afters[i])

        generate_fig(position_before, position_after, f'transitioning_into_state_{key}', window)
        plot_lines(position_before, position_after, window, key)
        position_before = defaultdict(list)
        position_after = defaultdict(list)        

def generate_fig(rt_before, rt_after, st, window):

    """"Initializes a figure for peristimulus histograms"""

    fig, axes = plt.subplots(nrows=1, ncols=len(rt_before)*2, figsize=(400, 50), dpi=80, facecolor='w', sharey=True, sharex=True, edgecolor='k')
    
    ind = 0
    for key in rt_before:
        #plot_trans_hists(axes, rt_before[key], rt_after[key], st, key, ind, ind+window, col1='tab:purple', col2 = 'tab:red')
        ind+=1

    plt.tight_layout()
    plt.savefig(f'peristim_hists/{experiment}/peristim_hist_drop_{experiment}_{st}.png')
    plt.close(fig) # close previous figure otherwise computer runs out of memory


def plot_lines(rct_before, rct_after, window, state):

    """Plots lines representing mean reaction times per position"""

    avrg_before = []
    avrg_after = []
    stdev_before = []
    stdev_after = []
    fig = plt.figure(figsize=(30, 10))
    #x1 = np.arange(-window, 0, 1)
    #x2 = np.arange(1, window+1, 1)
    x1 = ['pos -10', 'pos -9', 'pos -8', 'pos -7', 'pos -6', 'pos -5', 'pos -4', 'pos -3', 'pos -2', 'pos -1']
    x2 = ['pos 1', 'pos 2', 'pos 3', 'pos 4', 'pos 5', 'pos 6', 'pos 7', 'pos 8', 'pos 9', 'pos 10']
    positions1 = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]
    positions2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for key in rct_before:
        filt_rt_before = [x for x in rct_before[key] if not math.isnan(x)] # filter out nans
        filt_rt_after = [x for x in rct_after[key] if not math.isnan(x)] # filter out nans
        filt_rt_before = np.array(filt_rt_before)
        filt_rt_after = np.array(filt_rt_after)
        mask1 = filt_rt_before < 1250
        mask2 = filt_rt_after < 1250
        new_before = filt_rt_before[mask1]
        new_after = filt_rt_after[mask2]
        avrg_before.append(new_before)
        avrg_after.append(new_after)
    plt.boxplot(avrg_before, vert=True, patch_artist=True, meanline=True, positions=positions1, widths=0.3,boxprops=dict(linewidth=2), whiskerprops=dict(linewidth=2), capprops=dict(linewidth=2), meanprops=dict(marker='D', linestyle='--', markerfacecolor='red', linewidth=2))
    plt.boxplot(avrg_after, vert=True, patch_artist=True, meanline=True, positions=positions2, widths=0.3, boxprops=dict(facecolor='pink', linewidth=2), whiskerprops=dict(linewidth=2), capprops=dict(linewidth=2), meanprops=dict(marker='D', linestyle='--', markerfacecolor='blue', linewidth=2))

        # avrg_before.append(sum(filt_rt_before)/len(filt_rt_before))
        # avrg_after.append(sum(filt_rt_after)/len(filt_rt_after))
        # stdev_before.append(statistics.pstdev(filt_rt_before))
        # stdev_after.append(statistics.pstdev(filt_rt_after))
    # x1 = np.arange(-window, 0, 1)
    # x2 = np.arange(1, window+1, 1)
    # stdev_before = np.array(stdev_before)
    # stdev_after = np.array(stdev_after)
    # avrg_before = np.array(avrg_before)
    # avrg_after = np.array(avrg_after)
    #import ipdb; ipdb.set_trace()
    #plt.plot(x1, avrg_before, color = 'k')
    #plt.errorbar(x1, avrg_before, yerr=stdev_before, fmt='none', capsize=5, capthick=2, ecolor='green')
    #plt.fill_between(x1, avrg_before-stdev_before, avrg_before+stdev_after)
    #plt.plot(x2, avrg_after, color='r')
    #plt.errorbar(x2, avrg_after, yerr=stdev_after, fmt='none', capsize=5, capthick=2, ecolor='green')
    #plt.tight_layout()
    if state == 0:
        plt.title(f'transitioning into optimal state', fontsize=30)
    if state == 1:
        plt.title(f'transitioning into biased state', fontsize=30)
    if state == 2:
        plt.title(f'transitioning into no response state', fontsize=30)
    plt.xlabel('position', fontsize=30)
    plt.ylabel('reaction times', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.savefig(f'box_plots/{experiment}/box_fig_{state}_{experiment}')
    avrg_before = []
    avrg_after = []
    stdev_before = []
    stdev_after = [] 
  

def plot_trans_hists(axes, rct_before, rct_after, st, key, ind1, ind2, col1, col2):

    """Takes an array of reaction times and plots them on a histograms showing 
    before and after transition reaction times"""
    

    nan_count_before = np.sum(np.isnan(rct_before))
    nan_count_after = np.sum(np.isnan(rct_after))
    bins = np.arange(0, 2001, 100)
    axes[ind1].hist(rct_before, bins=bins, edgecolor='k', color=col1)
    axes[ind1].tick_params(axis='both', which='major', labelsize=80)
    axes[ind1].set_xlabel("reaction times", fontsize = 80)
    axes[ind1].set_ylabel("count", fontsize = 80)
    axes[ind1].set_title(f"RT before {st} nan # {nan_count_before} out {len(rct_before)} values", fontsize=50)
    filt_rt_before = [x for x in rct_before if not math.isnan(x)] # filter out nans
    avrg_before = sum(filt_rt_before)/len(filt_rt_before)
    axes[ind1].axvline(avrg_before, color='k', linestyle='dashed', linewidth=10)
    #min_ylim, max_ylim = axes[0].set_ylim()
    axes[ind1].text(avrg_before + 5, 8, 'Mean: {:.2f}'.format(avrg_before), fontsize=80)

    axes[ind2].hist(rct_after, bins=bins, edgecolor='k', color=col2)
    axes[ind2].tick_params(axis='both', which='major', labelsize=80)
    axes[ind2].set_xlabel("reaction times", fontsize = 80)
    axes[ind2].set_ylabel("count", fontsize = 80)
    axes[ind2].set_title(f"RT after {st} nan # {nan_count_after} out {len(rct_after)} values", fontsize=50)
    filt_rt_after = [x for x in rct_after if not math.isnan(x)] # filter out nans
    avrg_after = sum(filt_rt_after)/len(filt_rt_after)
    axes[ind2].axvline(avrg_after, color='k', linestyle='dashed', linewidth=10)
    axes[ind2].text(avrg_before + 5, 8, 'Mean: {:.2f}'.format(avrg_after), fontsize=80)




if __name__ == "__main__":
    args = get_args()
    choice_f: str = args.choices #holds path or file name of mouse choice data
    #input_f: str = args.inputs #holds path or file name of mouse inputs data
    stimulus_f: str = args.stimilus # holds a file name to mouse stimulus data
    react_times_f = args.react_times #holds path for reaction times
    experiment: str = args.gname #holds experiment number to use as a graph header
         #load mouse inputs and choices
    npr.seed(42)

    #inputs = np.load(input_f,allow_pickle = True) # load numpy file inputs
    

    choices = np.load(choice_f,allow_pickle = True) # load numpy files choices

    choices_exp = []
    for session in choices:
        expanded_arr = np.expand_dims(session, axis=1)
        choices_exp.append(expanded_arr)
    choices = np.array(choices_exp, dtype=object)
    stimulus = np.load(stimulus_f, allow_pickle=True) # load numpy stimulus
    # generate inputs from stimulus arrays by adding the second column containing value 1.
    inputs = []
    for session in stimulus:
        new_col = np.ones_like(session)
        result = np.column_stack((session, new_col))
        inputs.append(result)

    react_times = np.load(react_times_f, allow_pickle=True) # load numpy file reaction times
    react_times_exp = []
    for session in react_times:
        rt_exp = np.array(session)
        react_times_exp.append(rt_exp)
    react_times_exp = np.array(react_times_exp, dtype=object)

    input_dim: int = inputs[0].shape[1]  # input dimensions
    num_states = 3        # number of discrete states
    TOL: float = 10**-4 # tolerance 
    N_iters: int = 1000 # number of iterations for the fitting model

    obs_dim: int = choices_exp[0].shape[1]          # number of observed dimensions
    num_categories: int = len(np.unique(np.concatenate(choices)))    # number of categories for output
    #import ipdb; ipdb.set_trace()
    hmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")
    fit_glmhmm = fit_glm_hmm(hmm, choices_exp, inputs, N_iters,TOL)
    state_probs = get_state_probs(hmm, choices_exp, inputs)

    wind_sze = 5
    parse_probs(state_probs, react_times_exp, wind_sze)

    #import ipdb; ipdb.set_trace() 
    drop_trans = []
    raise_trans = []
    states_dict = defaultdict(list)
    for sess_id in range(len(state_probs)):

        max_probs = parse_probs_by_state(state_probs[sess_id])
        drop_trans, raise_trans = find_state_change_inds(max_probs, drop_trans, raise_trans)
        combine_probs_by_trans(drop_trans, raise_trans, states_dict, sess_id)
   
        drop_trans = []
        raise_trans = []
    get_rt_values(react_times_exp, states_dict, window=10)
    
    #import ipdb; ipdb.set_trace() 
    
    
    

