#!/usr/bin/env python

# The purpose of this modeule is to simulate psytrack weights using our experimental values of stimuli.
# the original psytrack simulation function draws stimulus data which is called inputs there from
# the standard random, and we want to make it in a control fashion with drawing from the following 
# value list: [-1, -0.5, -0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25, 0.5, 1] Note: the original list
# also contained value 0 which I had to filter anyway, and I am not including it here 

import numpy as np
from datetime import datetime, timedelta
from os import makedirs



def generateSimData(K=1,
                N=64000,
                hyper={},
                stim_list=None,
                days=None,
                boundary=4.0,
                iterations=20,
                seed=None,
                savePath=None):
    '''Simulates weights, inputs, and choices under the model.

    Args:
        K : int, number of weights to simulate
        N : int, number of trials to simulate
        hyper : dict, hyperparameters and initial values used to construct the
            prior. Default is none, can include sigma, sigInit, sigDay
        days : list or array, list of the trial indices on which to apply the
            sigDay hyperparameter instead of the sigma
        boundary : float, weights are reflected from this boundary
            during simulation, is a symmetric +/- boundary
        iterations : int, # of behavioral realizations to simulate,
            same input and weights can render different choice due
            to probabilistic model, iterations are saved in 'all_Y'
        seed : int, random seed to make random simulations reproducible
        savePath : str, if given creates a folder and saves simulation data
            in a file; else data is returned

    Returns:
        save_path | (if savePath) : str, the name of the folder+file where
            simulation data was saved in the local directory
        save_dict | (if no SavePath) : dict, contains all relevant info
            from the simulation 
    '''

    # Reproducability
    np.random.seed(seed)

    # Supply default hyperparameters if necessary
    sigmaDefault = 2**np.random.choice([-4.0, -5.0, -6.0, -7.0, -8.0], size=K)
    sigInitDefault = np.array([4.0] * K)
    sigDayDefault = 2**np.random.choice([1.0, 0.0, -1.0], size=K)

    if 'sigma' not in hyper:
        sigma = sigmaDefault
    elif hyper['sigma'] is None:
        sigma = sigmaDefault
    elif np.isscalar(hyper['sigma']):
        sigma = np.array([hyper['sigma']] * K)
    elif ((type(hyper['sigma']) in [np.ndarray, list]) and
          (len(hyper['sigma']) == K)):
        sigma = hyper['sigma']
    else:
        raise Exception('hyper["sigma"] must be either a scalar or a list or '
                        'array of len K')

    if 'sigInit' not in hyper:
        sigInit = sigInitDefault
    elif hyper['sigInit'] is None:
        sigInit = sigInitDefault
    elif np.isscalar(hyper['sigInit']):
        sigInit = np.array([hyper['sigInit']] * K)
    elif (type(hyper['sigInit']) in [np.ndarray, list]) and (len(hyper['sigInit']) == K):
        sigInit = hyper['sigInit']
    else:
        raise Exception('hyper["sigInit"] must be either a scalar or a list or '
                        'array of len K.')

    if days is None:
        sigDay = None
    elif 'sigDay' not in hyper:
        sigDay = sigDayDefault
    elif hyper['sigDay'] is None:
        sigDay = sigDayDefault
    elif np.isscalar(hyper['sigDay']):
        sigDay = np.array([hyper['sigDay']] * K)
    elif ((type(hyper['sigDay']) in [np.ndarray, list]) and
          (len(hyper['sigDay']) == K)):
        sigDay = hyper['sigDay']
    else:
        raise Exception('hyper["sigDay"] must be either a scalar or a list or '
                        'array of len K.')

    # -------------
    # Simulation
    # -------------

    # Simulate inputs
    #X = np.random.normal(size=(N, K))
    X = np.random.choice(stim_list, size=(N, K))
    # Simulate weights
    E = np.zeros((N, K))
    E[0] = np.random.normal(scale=sigInit, size=K)
    E[1:] = np.random.normal(scale=sigma, size=(N - 1, K))
    if sigDay is not None:
        E[np.cumsum(days)] = np.random.normal(scale=sigDay, size=(len(days), K))
    W = np.cumsum(E, axis=0)

    # Impose a ceiling and floor boundary on W
    for i in range(len(W.T)):
        cross = (W[:, i] < -boundary) | (W[:, i] > boundary)
        while cross.any():
            ind = np.where(cross)[0][0]
            if W[ind, i] < -boundary:
                W[ind:, i] = -2*boundary - W[ind:, i]
            else:
                W[ind:, i] = 2*boundary - W[ind:, i]
            cross = (W[:, i] < -boundary) | (W[:, i] > boundary)

    # Save data
    save_dict = {
        'sigInit': sigInit,
        'sigDay' : sigDay,
        'sigma': sigma,
        'dayLength' : days,
        'seed': seed,
        'W': W,
        'X': X,
        'K': K,
        'N': N,
    }

    # Simulate behavioral realizations in advance
    pR = 1.0 / (1.0 + np.exp(-np.sum(X * W, axis=1)))

    all_simy = []
    for i in range(iterations):
        sim_y = (pR > np.random.rand(
            len(pR))).astype(int) + 1  # 1 for L, 2 for R
        all_simy += [sim_y]

    # Update saved data to include behavior
    save_dict.update({'all_Y': all_simy})

    # Save & return file path OR return simulation data
    if savePath is not None:
        # Creates unique file name from current datetime
        folder = datetime.now().strftime('%Y%m%d_%H%M%S') + savePath
        makedirs(folder)

        fullSavePath = folder + '/sim.npz'
        np.savez_compressed(fullSavePath, save_dict=save_dict)

        return fullSavePath

    else:
        return save_dict
