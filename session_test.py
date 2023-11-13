#!/usr/bin/env python

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
    #parser.add_argument("-e", "--end", help="ending number of trials", required=True, type=int)
    #parser.add_argument("-i", "--increment", help="increments of trials", required=True, type=int)
    parser.add_argument("-se", "--session", help="total number of sessions", required=True, type=int)
    parser.add_argument("-o", "--outfile", help="output file name", required=True, type=str)
    return parser.parse_args()

args = get_args()
start = args.start
#end = args.end  # Use the 'end' argument to specify the ending number of trials
#inc = args.increment
sess = args.session #total number of sessions
out = args.outfile

# Set the parameters of the GLM-HMM
num_states = 3        # number of discrete states
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output
input_dim = 2         # input dimensions

# Make a GLM-HMM
true_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")

gen_weights = np.array([[[6, 1]], [[2, -3]], [[2, 3]]])
gen_log_trans_mat = np.log(np.array([[[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.03, 0.03, 0.94]]]))
true_glmhmm.observations.params = gen_weights
true_glmhmm.transitions.params = gen_log_trans_mat

# # Plot generative parameters:
# fig = plt.figure(figsize=(8, 3), dpi=80, facecolor='w', edgecolor='k')
# plt.subplot(1, 2, 1)
cols = ['#ff7f00', '#4daf4a', '#377eb8']
# for k in range(num_states):
#     plt.plot(range(input_dim), gen_weights[k][0], marker='o',
#              color=cols[k], linestyle='-',
#              lw=1.5, label="state " + str(k+1))
# plt.yticks(fontsize=10)
# plt.ylabel("GLM weight", fontsize=15)
# plt.xlabel("covariate", fontsize=15)
# plt.xticks([0, 1], ['stimulus', 'bias'], fontsize=12, rotation=45)
# plt.axhline(y=0, color="k", alpha=0.5, ls="--")
# plt.legend()
# plt.title("Generative weights", fontsize=15)

# plt.subplot(1, 2, 2)
# gen_trans_mat = np.exp(gen_log_trans_mat)[0]
# plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
# for i in range(gen_trans_mat.shape[0]):
#     for j in range(gen_trans_mat.shape[1]):
#         text = plt.text(j, i, str(np.around(gen_trans_mat[i, j], decimals=2)), ha="center", va="center",
#                         color="k", fontsize=12)
# plt.xlim(-0.5, num_states - 0.5)
# plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
# plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
# plt.ylim(num_states - 0.5, -0.5)
# plt.ylabel("state t", fontsize = 15)
# plt.xlabel("state t+1", fontsize = 15)
# plt.title("Generative transition matrix", fontsize = 15)
# plt.savefig(f'{out}') 

# Simulation loop
# Simulation loop
for j in range(1, sess+1):  # Run simulations for different sets start, end + 1, inc
    num_sess = j  # number of example sessions (set to 10)
    num_trials_per_sess = start  # number of trials in a session
    inpts = np.ones((num_sess, num_trials_per_sess, input_dim))  # initialize inpts array
    stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]
    inpts[:, :, 0] = np.random.choice(stim_vals, (num_sess, num_trials_per_sess))  # generate random sequence of stimuli
    inpts = list(inpts)  # convert inpts to correct format

    true_latents, true_choices = [], []
    for sess in range(num_sess):
        true_z, true_y = true_glmhmm.sample(num_trials_per_sess, input=inpts[sess])
        true_latents.append(true_z)
        true_choices.append(true_y)

    true_ll = true_glmhmm.log_probability(true_choices, inputs=inpts) 
    print(f"Set {j}: true ll = {true_ll}")

    new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                    observation_kwargs=dict(C=num_categories), transitions="standard")

    N_iters = 200  # maximum number of EM iterations. Fitting will stop earlier if the increase in LL is below the tolerance specified by the tolerance parameter
    fit_ll = new_glmhmm.fit(true_choices, inputs=inpts, method="em", num_iters=N_iters, tolerance=10**-4)
    #new_glmhmm.permute(find_permutation(true_latents[0], new_glmhmm.most_likely_states(true_choices[0], input=inpts[0])))
    print(f"Set {j}: Fitted LL = {fit_ll}")


    # Get expected states:
    posterior_probs = [new_glmhmm.expected_states(data=data, input=inpt)[0]
                for data, inpt
                in zip(true_choices, inpts)]
    fig = plt.figure(figsize=(10, 2.5), dpi=80, facecolor='w', edgecolor='k')
    sess_id = num_sess #session id; can choose any index between 0 and num_sess-1
    for k in range(num_states):
        plt.plot(posterior_probs[sess_id][:, k], label="State " + str(k + 1), lw=2,
                color=cols[k])
    plt.ylim((-0.01, 1.01))
    plt.yticks([0, 0.5, 1], fontsize = 10)
    plt.xlabel("trial #", fontsize = 15)
    plt.ylabel("p(state)", fontsize = 15)
    plt.savefig(f'states_fig{out}session{num_sess}')
    # fig = plt.figure(figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    # cols = ['#ff7f00', '#4daf4a', '#377eb8']
    # recovered_weights = new_glmhmm.observations.params
    # print(recovered_weights)
    # for k in range(num_states):
    #     if k == 0:
    #         plt.plot(range(input_dim), gen_weights[k][0], marker='o',
    #                 color=cols[k], linestyle='-',
    #                 lw=1.5, label="generative")
    #         plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
    #                     lw=1.5,  label="recovered", linestyle='--')
    #     else:
    #         plt.plot(range(input_dim), gen_weights[k][0], marker='o',
    #                 color=cols[k], linestyle='-',
    #                 lw=1.5, label="")
    #         plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
    #                     lw=1.5, label="", linestyle='--')
    # plt.yticks(fontsize=10)
    # plt.ylabel("GLM weight", fontsize=15)
    # plt.xlabel("covariate", fontsize=15)
    # plt.xticks([0, 1], ['stimulus', 'bias'], fontsize=12, rotation=45)
    # plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    # plt.legend()
    # plt.title(f"Weight recovery (Set {j})", fontsize=15)
    # plt.savefig(f'{out}_{j}.png')
