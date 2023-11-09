import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import ssm
from ssm.util import find_permutation
import argparse

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
    return true_glmhmm

def generate_inputs(num_trials_per_sess):

    """Takes """

    #num_sess = sess  # number of example sessions (set to 10)
    #num_trials_per_sess = j  # number of trials in a session
    inpts = np.ones((num_sess, num_trials_per_sess, input_dim))  # initialize inpts array
    stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]
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

    # set weights and transition matrix numbers for the HMM model initiated with specific starting
    #  parameters
    gen_weights = np.array([[[6, 1]], [[2, -3]], [[2, 3]]])
    gen_log_trans_mat = np.log(np.array([[[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.03, 0.03, 0.94]]]))
    # initialize an glmhmm with specific parameters
    true_glmhmm = make_standard_hmm(gen_weights, gen_log_trans_mat)

    for j in range(start, end + 1, inc):
        print(f'{num_sess=}')
        print(f'{j}')
        print(f'{inc=}')
        inpts=generate_inputs(j)
        true_ll, true_latents, true_choices = true_ll_model(j, num_sess)

        # inialize new hmm without standard input values
        new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                            observation_kwargs=dict(C=num_categories), transitions="standard")
            
        fit_ll = fit_glm_hmm(new_glmhmm)
    