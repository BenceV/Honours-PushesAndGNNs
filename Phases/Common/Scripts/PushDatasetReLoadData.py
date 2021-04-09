
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


#Import graph nets
from graph_nets import blocks
from graph_nets import utils_tf
from graph_nets import utils_np


def get_data(dfs, tr_batch_size, va_batch_size, te_batch_size):
    
    result = {}
    
    # Get training, validation, testing examples
    tr_data, result["tr_indeces"] = get_batch_of_states(dfs["tr_1"], dfs["tr_2"], tr_batch_size)
    va_data, result["va_indeces"] = get_batch_of_states(dfs["va_1"], dfs["va_2"], va_batch_size)
    te_data, result["te_indeces"] = get_batch_of_states(dfs["te_1"], dfs["te_2"], te_batch_size)
    
    # Convert Training data
    X_dicts = [rigid_graph_from_pos_all(obj_pos, tip_pos, tip_vel, tip_con) for obj_pos, tip_pos, tip_vel, tip_con, in zip(tr_data['obj_pos_X'], tr_data['tip_pos_X'], tr_data['tip_vel_X'], tr_data['tip_con_X'])]
    Y_dicts = [rigid_graph_from_pos_all(obj_pos, tip_pos, tip_vel, tip_con) for obj_pos, tip_pos, tip_vel, tip_con, in zip(tr_data['obj_pos_Y'], tr_data['tip_pos_Y'], tr_data['tip_vel_Y'], tr_data['tip_con_Y'])]
    
    result["X_tr"] = utils_tf.data_dicts_to_graphs_tuple(X_dicts)
    result["Y_tr"] = utils_tf.data_dicts_to_graphs_tuple(Y_dicts)
    
    # Convert Validation data
    X_dicts = [rigid_graph_from_pos_all(obj_pos, tip_pos, tip_vel, tip_con) for obj_pos, tip_pos, tip_vel, tip_con, in zip(va_data['obj_pos_X'], va_data['tip_pos_X'], va_data['tip_vel_X'], va_data['tip_con_X'])]
    Y_dicts = [rigid_graph_from_pos_all(obj_pos, tip_pos, tip_vel, tip_con) for obj_pos, tip_pos, tip_vel, tip_con, in zip(va_data['obj_pos_Y'], va_data['tip_pos_Y'], va_data['tip_vel_Y'], va_data['tip_con_Y'])]
    
    result["X_va"] = utils_tf.data_dicts_to_graphs_tuple(X_dicts)
    result["Y_va"] = utils_tf.data_dicts_to_graphs_tuple(Y_dicts)
    
    # Convert Testing data
    X_dicts = [rigid_graph_from_pos_all(obj_pos, tip_pos, tip_vel, tip_con) for obj_pos, tip_pos, tip_vel, tip_con, in zip(te_data['obj_pos_X'], te_data['tip_pos_X'], te_data['tip_vel_X'], te_data['tip_con_X'])]
    Y_dicts = [rigid_graph_from_pos_all(obj_pos, tip_pos, tip_vel, tip_con) for obj_pos, tip_pos, tip_vel, tip_con, in zip(te_data['obj_pos_Y'], te_data['tip_pos_Y'], te_data['tip_vel_Y'], te_data['tip_con_Y'])]
    
    result["X_te"] = utils_tf.data_dicts_to_graphs_tuple(X_dicts)
    result["Y_te"] = utils_tf.data_dicts_to_graphs_tuple(Y_dicts)
    
    return result

def get_data_for_training(epochs, dfs, tr_batch_size, va_batch_size, te_batch_size):
    data = []
    for i in range(epochs):
        d = get_data(dfs, tr_batch_size, va_batch_size, te_batch_size)
        data.extend([d])
    return np.array(data)

def reload_data():
    # Get data
    if False:
        data_path = os.path.join(path_saves, "data_"+str(SEED)+"n_steps_"+str(NUM_TRAINING_ITERATIONS)+".npy")
        if not os.path.exists(data_path):
            print("Collecting ...")
            # If data with same seed exists, load it, else, create it
            data_for_training = get_data_for_training(epochs = NUM_TRAINING_ITERATIONS, 
                                                    dfs = df_dict, 
                                                    tr_batch_size = BATCH_SIZE_TR, 
                                                    va_batch_size = BATCH_SIZE_GE, 
                                                    te_batch_size = BATCH_SIZE_TE)
            # Save data with for
            np.save(open(data_path, 'wb'),data_for_training, allow_pickle=True)

        else:
            print("Loading ...")
            # Load data
            data_for_training = np.load(open(data_path, 'rb'), allow_pickle=True)
