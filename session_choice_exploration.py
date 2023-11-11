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

    """Takes """

    #num_sess = sess  # number of example sessions (set to 10)
    #num_trials_per_sess = j  # number of trials in a session
    print(f'generate ipts{num_trials_per_sess=}')
    inpts = np.ones((num_sess, num_trials_per_sess, input_dim))  # initialize inpts array
    #stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]
    inpts[:, :, 0] = np.random.choice(stim_vals, (num_sess, num_trials_per_sess))  # generate random sequence of stimuli
    inpts = list(inpts)  # convert inpts to correct format
    return inpts

def true_ll_model(num_trials_per_sess, num_sess):

    """"Takes""" 
    true_latents, true_choices = [], []
    for sess in range(num_sess):
        true_z, true_y = true_glmhmm.sample(num_trials_per_sess, input=inpts[sess])
        true_latents.append(true_z)
        true_choices.append(true_y)

    true_ll = true_glmhmm.log_probability(true_choices, inputs=inpts)
    return true_ll, true_latents, true_choices

def fit_glm_hmm(new_glmhmm):

    """Takes"""

    fit_ll = new_glmhmm.fit(true_choices, inputs=inpts, method="em", num_iters=N_iters, tolerance=10**-4)

    return fit_ll


def plot_weights(new_glmhmm):

    """Takes """

    fig = plt.figure(figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    cols = ['#ff7f00', '#4daf4a', '#377eb8']
    recovered_weights = new_glmhmm.observations.params
    #print(recovered_weights)
    for k in range(num_states):
        if k == 0:
            plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                    color=cols[k], linestyle='-',
                    lw=1.5, label="generative")
            plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                        lw=1.5,  label="recovered", linestyle='--')
        else:
            plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                    color=cols[k], linestyle='-',
                    lw=1.5, label="")
            plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                        lw=1.5, label="", linestyle='--')
    plt.yticks(fontsize=10)
    plt.ylabel("GLM weight", fontsize=15)
    plt.xlabel("covariate", fontsize=15)
    plt.xticks([0, 1], ['stimulus', 'bias'], fontsize=12, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.legend()
    plt.title(f"Weight recovery (Set {j})", fontsize=15)
    #plt.savefig(f'{out}_{j}.png')
    return fig

def plot_states(new_glmhmm):

    """Takes"""

    cols = ['#ff7f00', '#4daf4a', '#377eb8']
    posterior_probs = [new_glmhmm.expected_states(data=data, input=inpt)[0]
                for data, inpt
                in zip(true_choices, inpts)]
    fig = plt.figure(figsize=(15, 2.5), dpi=80, facecolor='w', edgecolor='k')
    sess_id = 0 #session id; can choose any index between 0 and num_sess-1
    for k in range(num_states):
        plt.plot(posterior_probs[sess_id][:, k], label="State " + str(k + 1), lw=2,
                color=cols[k])
    plt.ylim((-0.01, 1.01))
    plt.yticks([0, 0.5, 1], fontsize = 10)
    plt.xlabel("trial #", fontsize = 15)
    plt.ylabel("p(state)", fontsize = 15)
    #plt.savefig(f'states_fig{out}session{j}.png')
    return fig

def plot_choices(n_sess):

    """Takes"""

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
    fig = plt.figure(figsize=(15, 2.5), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.axes()
    x_range = np.array(range(len(cho)))
    scatter_correct = plt.scatter(x_range[correct_choices], y_values_jittered[correct_choices], 
                                  label='correct', color='r', alpha=alpha)
    scatter_wrong = plt.scatter(x_range[~correct_choices], y_values_jittered[~correct_choices],
                                 label='wrong', color='k',alpha=alpha)
    ax.set_yticks([0,1], ['L', 'R'])
    plt.legend()
    plt.xlabel("session #", fontsize = 15)
    plt.ylabel("choice", fontsize = 15)
    #plt.savefig(f'choices_fig{out}session{j}.png')
    return fig
    


if __name__ == "__main__":
    args = get_args()
    start = args.start
    end = args.end  # Use the 'end' argument to specify the ending number of trials
    inc = args.increment
    num_sess = args.session
    out = args.outfile

    # Set the parameters of the GLM-HMM
    num_states = 3        # number of discrete states
    obs_dim = 1           # number of observed dimensions
    num_categories = 2    # number of categories for output
    input_dim = 2         # input dimensions
    N_iters = 1000          # number of fit iterations 
    # set stimulus values negative value correct choice is left [0], positive value correct choice 
    # is right[1]
    stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]
    # set weights and transition matrix numbers for the HMM model initiated with specific starting
    #  parameters
    gen_weights = np.array([[[6, 1]], [[2, -3]], [[2, 3]]])
    gen_log_trans_mat = np.log(np.array([[[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.03, 0.03, 0.94]]]))
    # initialize an glmhmm with specific parameters
    true_glmhmm = make_standard_hmm(gen_weights, gen_log_trans_mat)

    

    for j in range(start, end + 1, inc):
        
        inpts=generate_inputs(j)
        
        true_ll, true_latents, true_choices = true_ll_model(j, num_sess)
        
        #import ipdb; ipdb.set_trace()
        # inialize new hmm without standard input values
        new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                            observation_kwargs=dict(C=num_categories), transitions="standard")
        #new_glmhmm.permute(find_permutation(true_latents[0], new_glmhmm.most_likely_states(true_choices[0], input=inpts[0])))    
        fit_ll = fit_glm_hmm(new_glmhmm)

        plot_weights(new_glmhmm)
           
        plot_states(new_glmhmm)
           
        plot_choices(j)
        
        break
