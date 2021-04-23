from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import datetime
import json

#Import usual things required for graph nets
import numpy as np
import pandas as pd
import networkx as nx
import sonnet as snt
import tensorflow as tf
import os
import sys
import re

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon

import seaborn as sns

def rollout_plot(steps, rollout_error, path_plots, metric = "Position", title = "Rollout Plot"):
    fig = plt.figure(1, figsize=(10, 5))
    
    ax = fig.add_subplot(1, 1, 1)

    x = np.array(steps)
    y = np.mean(rollout_error,axis=-1).squeeze()
    e = np.std(rollout_error,axis=-1).squeeze()

    ax.plot(x, y, "orange",label = "Average")
    plt.fill_between(x, y-e, y+e, color = "orange", alpha=0.4, label = "Variance")

    ax.set_xlabel('Rollout Steps')
    ax.set_ylabel(str(metric)+' Error')
    ax.set_title(title)
    
    
    #Save figure
    name = re.split(' |: ', title)
    file_name = metric+"".join(name)+".png"
    if os.path.exists(os.path.join(path_plots, file_name)):
        print("The file: "+ file_name + " already exists. Delete it before saving a new plot!")
    else:
        if not os.path.exists(os.path.join(path_plots)):
            os.mkdir(os.path.join(path_plots))
        fig.savefig(os.path.join(path_plots, file_name))

    return fig, ax


def rollout_plot_log_scale(steps, rollout_error, path_plots, metric = "Position", title = "Rollout Plot: Log Scale"):
    fig = plt.figure(1, figsize=(10, 5))
    
    ax = fig.add_subplot(1, 1, 1)

    x = np.array(steps)
    y = np.mean(np.log(rollout_error),axis=-1).squeeze()
    e = np.std(np.log(rollout_error),axis=-1).squeeze()
    
    ax.plot(x, y, "orange",label = "Average")
    plt.fill_between(x, y-e, y+e, color = "orange", alpha=0.4, label = "Variance")
    
    ax.set_xlabel('Rollout Steps')
    ax.set_ylabel(str(metric)+' Error: Log Scale')
    ax.set_title(title)

    ax.legend()

    #Save figure
    name = re.split(' |: ', title)
    file_name = metric+"".join(name)+".png"
    if os.path.exists(os.path.join(path_plots, file_name)):
        print("The file: "+ file_name + " already exists. Delete it before saving a new plot!")
    else:
        if not os.path.exists(os.path.join(path_plots)):
            os.mkdir(os.path.join(path_plots))
        fig.savefig(os.path.join(path_plots, file_name))

    return fig, ax