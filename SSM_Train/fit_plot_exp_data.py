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


def get_args():

    """This grabs arguments for setting total trials"""

    parser = argparse.ArgumentParser(description="global variables to set input file names")
    parser.add_argument("-ch", "--choices", help="choices file name", required=True, type=str)
    parser.add_argument("-ip", "--inputs", help="input file name", required=True, type=str)
    parser.add_argument("-g", "--gname", help="graph name", required=True, type=str)
    return parser.parse_args()

if __name__ == "__main__":
    