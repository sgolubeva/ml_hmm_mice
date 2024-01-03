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

if __name__ == "__main__":
    args = get_args()
    react_times_f = args.react_times #holds path for reaction times
    experiment: str = args.gname #holds experiment number to use as a graph header
    npr.seed(42)