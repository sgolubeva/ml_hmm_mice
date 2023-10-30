## load data and fit glm-hmm
import numpy as np
import ssm

inpts = np.load(r'D:\Hulsey\BW051_behavior_data\BW051_inpts.npy',allow_pickle = True)
choices = np.load(r'D:\Hulsey\BW051_behavior_data\BW051_choices.npy',allow_pickle = True)

obs_dim = choices[0].shape[1]          # number of observed dimensions
num_categories = len(np.unique(np.concatenate(choices)))    # number of categories for output
input_dim = inpts[0].shape[1]                                    # input dimensions
num_states = 3

TOL = 10**-4

N_iters = 1000

hmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")

train_ll = hmm.fit(np.concatenate(choices), inputs=np.concatenate(inpts), method="em", num_iters=N_iters, tolerance=TOL)   

# get log likelihood of data set not used to fit model
# test_ll = hmm.log_probability(test_choices,test_inpts)

#### kfolds
from sklearn.model_selection import KFold
nKfold=5
kf = KFold(n_splits=nKfold, shuffle=True, random_state=None)
#Just for sanity's sake, let's check how it splits the data
for ii, (train_index, test_index) in enumerate(kf.split(choices)):
    print(f"kfold {ii} TRAIN:", len(train_index), "TEST:", len(test_index))


## using ray to parallelize
## https://www.anyscale.com/blog/writing-your-first-distributed-python-application-with-ray

