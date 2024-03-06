#!/usr/bin/env python3
"""The purpose of this script is to read a tab separated file, save result in two dictionaries
and plot results on a scatter plot. The output of train script is a tab separated file and this script
is designed to plot data to separate training from plotting"""

import matplotlib.pyplot as plt
import argparse
import numpy as np

def get_args():
    """Sets the command line arguments to run this script"""
    parser = argparse.ArgumentParser(description="global variables to set input file names")
    parser.add_argument("-it1", "--test", help="input test file", required=True, type=str)
    parser.add_argument("-it2", "--train", help="input train file", required=True, type=str)
    parser.add_argument("-e", "--exp", help="experiment name for graph name", required=True, type=str)
    return parser.parse_args()

args = get_args()

input_test: str = args.test # holds test log likelihood data
input_train: str = args.train # holds train log likelihood data
experiment: str = args.exp # holds graph name

test_likeli_dict = {}
train_likeli_dict = {}
with open(input_test) as fh, open(input_train) as ds:
        while True:
            test_line = fh.readline()
            train_line = ds.readline()
            if test_line == '':
                break
            test_line = test_line.strip('\n')
            train_line = train_line.strip('\n')
            test_line_spl = test_line.split(',')
            train_line_spl = train_line.split(',')   
            test_likeli_dict[float(test_line_spl[0])] = [float(test_line_spl[1]), float(test_line_spl[2]),
                                                      float(test_line_spl[3]), float(test_line_spl[4]),
                                                        float(test_line_spl[5])]
            train_likeli_dict[float(train_line_spl[0])] = [float(train_line_spl[1]), float(train_line_spl[2]),
                                                      float(train_line_spl[3]), float(train_line_spl[4]),
                                                        float(train_line_spl[5])]

keys = list(test_likeli_dict.keys())
values = list(test_likeli_dict.values())

x1 = []
y1 = []
x2 = []
y2 = []

fig, axs = plt.subplots(1,1)
fig.text(0.5,0.04, "State number n", ha="center", va="center", fontsize=15)
fig.text(0.02,0.5, "Log Likelihood", ha="center", va="center", rotation=90, fontsize=15)
for key in keys:
    x1.extend([key])
    y1.append(np.mean(test_likeli_dict[key]))

    x2.extend([key])
    y2.append(np.mean(train_likeli_dict[key]))

axs.scatter(x1, y1, label='test_data', color='indigo', s=100, alpha=0.8)
#axs[1].scatter(x2, y2, label='train_data', color='red')
#axs[1].yaxis.tick_right()
#axs[1].yaxis.set_ticks_position('both')
axs.yaxis.set_ticks_position('both')
plt.tick_params(axis='both', which='major', labelsize=9)
axs.set_title(f"Log_likelihood_{experiment}", fontsize=15)

plt.subplots_adjust(wspace=0, hspace=0)
# axs[0].legend(loc='lower right')
# axs[1].legend(loc='lower right')
plt.savefig(f'LL_{experiment}.png')
