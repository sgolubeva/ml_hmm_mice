#!/usr/bin/env python
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
import argparse
import numpy.random as npr


def get_args():

    """This grabs arguments for setting total trials"""

    parser = argparse.ArgumentParser(description="global variables to set input file names")
    #parser.add_argument("-ch", "--choices", help="choices file name", required=True, type=str)
    #parser.add_argument("-ip", "--inputs", help="input file name", required=True, type=str)
    parser.add_argument("-rt", "--react_times", help="reaction times file name", required=True, type=str)
    parser.add_argument("-g", "--gname", help="graph name", required=True, type=str)
    return parser.parse_args()

def baesian_fit(reaction_times, sess_id, experiment):

    """Takes a single session, fits it to the model, and generates a graph of transitions"""

    model = "rbf"  # Choose a model ("rbf" for Radial Basis Function)
    algo = rpt.Binseg(model=model).fit(reaction_times)
    result = rpt.Pelt(model=model).fit(reaction_times)
    result = algo.predict(pen=0)

    # plot
    rpt.display(reaction_times, result, figsize=(10, 6))
    plt.title(f'Change Point Detection using Pelt {experiment} session {sess_id}')
    plt.savefig(f'Change Point Detection using Pelt {experiment} session {sess_id}')
    #plt.close(fig) # close previous figure otherwise computer runs out of memory

if __name__ == "__main__":
    args = get_args()
    react_times_f = args.react_times #holds path for reaction times
    experiment: str = args.gname #holds experiment number to use as a graph header
    npr.seed(42)
    react_times = np.load(react_times_f, allow_pickle = True) # load numpy file reaction times
    
    for sess_id in range(len(react_times)) :
        baesian_fit(react_times[sess_id], sess_id, experiment)
