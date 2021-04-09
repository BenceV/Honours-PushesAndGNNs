
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

class DataFeeder:
    def __init__(self, SEED):
        np.random.seed(SEED)

    def get_data(self, dfs, tr_batch_size, va_batch_size, te_batch_size, graph_creator, te_and_va = True):
    
        result = {}

        # Get training, validation, testing examples
        tr_data, result["tr_indeces"] = self.get_batch_of_states(dfs["tr_1"], dfs["tr_2"], tr_batch_size)
        # Convert Training data
        X_dicts = [graph_creator(obj_pos, obj_vel, tip_pos, tip_vel, tip_con) for obj_pos, obj_vel, tip_pos, tip_vel, tip_con, in zip(tr_data['obj_pos_X'], tr_data['obj_vel_X'], tr_data['tip_pos_X'], tr_data['tip_vel_X'], tr_data['tip_con_X'])]
        Y_dicts = [graph_creator(obj_pos, obj_vel, tip_pos, tip_vel, tip_con) for obj_pos, obj_vel, tip_pos, tip_vel, tip_con, in zip(tr_data['obj_pos_Y'], tr_data['obj_vel_Y'], tr_data['tip_pos_Y'], tr_data['tip_vel_Y'], tr_data['tip_con_Y'])]
        
        result["X_tr"] = utils_tf.data_dicts_to_graphs_tuple(X_dicts)
        result["Y_tr"] = utils_tf.data_dicts_to_graphs_tuple(Y_dicts)

        if te_and_va:
            va_data, result["va_indeces"] = self.get_batch_of_states(dfs["va_1"], dfs["va_2"], va_batch_size)
            # Convert Validation data
            X_dicts = [graph_creator(obj_pos, obj_vel, tip_pos, tip_vel, tip_con) for obj_pos, obj_vel, tip_pos, tip_vel, tip_con, in zip(va_data['obj_pos_X'], va_data['obj_vel_X'], va_data['tip_pos_X'], va_data['tip_vel_X'], va_data['tip_con_X'])]
            Y_dicts = [graph_creator(obj_pos, obj_vel, tip_pos, tip_vel, tip_con) for obj_pos, obj_vel, tip_pos, tip_vel, tip_con, in zip(va_data['obj_pos_Y'], va_data['obj_vel_Y'], va_data['tip_pos_Y'], va_data['tip_vel_Y'], va_data['tip_con_Y'])]
            
            result["X_va"] = utils_tf.data_dicts_to_graphs_tuple(X_dicts)
            result["Y_va"] = utils_tf.data_dicts_to_graphs_tuple(Y_dicts)
            
            te_data, result["te_indeces"] = self.get_batch_of_states(dfs["te_1"], dfs["te_2"], te_batch_size)
            # Convert Testing data
            X_dicts = [graph_creator(obj_pos, obj_vel, tip_pos, tip_vel, tip_con) for obj_pos, obj_vel, tip_pos, tip_vel, tip_con, in zip(te_data['obj_pos_X'], te_data['obj_vel_X'], te_data['tip_pos_X'], te_data['tip_vel_X'], te_data['tip_con_X'])]
            Y_dicts = [graph_creator(obj_pos, obj_vel, tip_pos, tip_vel, tip_con) for obj_pos, obj_vel, tip_pos, tip_vel, tip_con, in zip(te_data['obj_pos_Y'], te_data['obj_vel_Y'], te_data['tip_pos_Y'], te_data['tip_vel_Y'], te_data['tip_con_Y'])]
            
            result["X_te"] = utils_tf.data_dicts_to_graphs_tuple(X_dicts)
            result["Y_te"] = utils_tf.data_dicts_to_graphs_tuple(Y_dicts)
        
        
        
        
        
    
        return result
 
    def convert_state(self, df): 
        obj_pos = df[["o_t_r_x", 
                    "o_t_r_y",
                    "o_t_l_x",
                    "o_t_l_y",
                    "o_b_r_x",
                    "o_b_r_y",
                    "o_b_l_x",
                    "o_b_l_y",
                    "o_m_m_x",
                    "o_m_m_y"]]
        
        obj_vel = df[["o_t_r_x_v", 
                    "o_t_r_y_v",
                    "o_t_l_x_v",
                    "o_t_l_y_v",
                    "o_b_r_x_v",
                    "o_b_r_y_v",
                    "o_b_l_x_v",
                    "o_b_l_y_v",
                    "o_m_m_x_v",
                    "o_m_m_y_v"]]
        
        tip_pos = df[["e_pos_x",
                    "e_pos_y"]]
        
        tip_vel = df[["e_vel_x",
                    "e_vel_y"]]
                
        tip_con = df[["contact"]]
        
        obj_pos = np.reshape(obj_pos.to_numpy(), (df.shape[0], 5, 2))
        obj_vel = np.reshape(obj_vel.to_numpy(), (df.shape[0], 5, 2))
        tip_pos = tip_pos.to_numpy()
        tip_vel = tip_vel.to_numpy()
        tip_con = tip_con.to_numpy()
        
        return obj_pos, obj_vel, tip_pos, tip_vel, tip_con



    def get_batch_of_states(self, df1, df2, batch_size):
        cols = [
            "o_t_r_x", 
            "o_t_r_y",
            "o_t_l_x",
            "o_t_l_y",
            "o_b_r_x",
            "o_b_r_y",
            "o_b_l_x",
            "o_b_l_y",
            "o_m_m_x",
            "o_m_m_y",
            "e_pos_x",
            "e_pos_y",
            "o_t_r_x_v", 
            "o_t_r_y_v",
            "o_t_l_x_v",
            "o_t_l_y_v",
            "o_b_r_x_v",
            "o_b_r_y_v",
            "o_b_l_x_v",
            "o_b_l_y_v",
            "o_m_m_x_v",
            "o_m_m_y_v",
            "e_vel_x",
            "e_vel_y",
            "contact"]
        
        result = {}
        h = df1.index.to_numpy()[-1]
            
        indeces = np.random.randint(low=0, high=h+1, size=batch_size)
        
        df_batch_1 = df1.iloc[indeces]
        df_batch_2 = df2.iloc[indeces]
        
        df_batch_1 = df_batch_1[cols]
        df_batch_2 = df_batch_2[cols]
        
        result["obj_pos_X"], result["obj_vel_X"], result["tip_pos_X"], result["tip_vel_X"], result["tip_con_X"] = self.convert_state(df_batch_1)
        result["obj_pos_Y"], result["obj_vel_Y"], result["tip_pos_Y"], result["tip_vel_Y"], result["tip_con_Y"] = self.convert_state(df_batch_2)
        
        return result, indeces
