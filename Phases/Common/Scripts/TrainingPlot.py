
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

def training_plot(logged_iterations, tr_loss, va_loss, te_loss, path_plots, loss = "Step", title = "Convergence Plot"):
    fig = plt.figure(1, figsize=(10, 5))
    
    ax = fig.add_subplot(1, 1, 1)

    x = np.array(logged_iterations)
    y = np.array(tr_loss)
    ax.plot(x, y, "orange", label="Training")
    y = np.array(va_loss)
    ax.plot(x, y, "blue", label="Validation")
    y = np.array(te_loss)
    ax.plot(x, y, "green", label="Testing")

    ax.set_xlabel('Iterations')
    ax.set_ylabel(str(loss)+' loss')
    ax.set_title(title)
    
    ax.legend()
    
    #Save figure
    name = re.split(' |: ', title)
    file_name = loss+"".join(name)+".png"
    if os.path.exists(os.path.join(path_plots, file_name)):
        print("The file: "+ file_name + "already exists. Delete it before saving a new plot!")
    else:
        if not os.path.exists(os.path.join(path_plots)):
            os.mkdir(os.path.join(path_plots))
        fig.savefig(os.path.join(path_plots, file_name))

    return fig, ax


def training_plot_log_scale(logged_iterations, tr_loss, va_loss, te_loss, path_plots, loss = "Step", title = "Convergence Plot: Log Scale"):
    fig = plt.figure(1, figsize=(10, 5))
    
    ax = fig.add_subplot(1, 1, 1)

    x = np.array(logged_iterations)
    
    y = np.log(tr_loss)
    ax.plot(x, y, "orange", label="Training")
    y = np.log(va_loss)
    ax.plot(x, y, "blue", label="Validation")
    y = np.log(te_loss)
    ax.plot(x, y, "green", label="Testing")
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel(str(loss) +' loss')
    ax.set_title(title)

    ax.legend()

    #Save figure
    name = re.split(' |: ', title)
    file_name =  loss+"".join(name)+".png"
    if os.path.exists(os.path.join(path_plots, file_name)):
        print("The file: "+ file_name + "already exists. Delete it before saving a new plot!")
    else:
        if not os.path.exists(os.path.join(path_plots)):
            os.mkdir(os.path.join(path_plots))
        fig.savefig(os.path.join(path_plots, file_name))

    return fig, ax