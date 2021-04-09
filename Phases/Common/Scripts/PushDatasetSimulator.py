
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




class PushDatasetSimulator:



    def __init__(self, rollout_steps, step_size):
        self.ROLLOUT_TIMESTEPS = rollout_steps
        self.STEP_SIZE = step_size

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

    def get_batch_of_trajectories(self, df1, df2, batch_size, trajectory_ids = None):
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

        ret = []
        internal_inds = {}
        traj_inds = df1.trajectory.unique()
        if trajectory_ids is not None:
            traj_inds = trajectory_ids
        else:
            siz = batch_size
            if batch_size > len(traj_inds):
                siz = len(traj_inds)
            traj_inds = traj_inds[np.random.choice(len(traj_inds), size=siz, replace=False)]    
        
        df_1 = df1.set_index("trajectory", drop=False)
        df_2 = df2.set_index("trajectory", drop=False)
        
        to_del = []
        for index, i in zip(traj_inds, range(len(traj_inds))):
            result = {} 
            a1 = df_1.loc[[index]]    
            a2 = df_2.loc[[index]]
            if a1.shape[0] > self.ROLLOUT_TIMESTEPS:
                # Get only the window slice
                h = len(a1) - self.ROLLOUT_TIMESTEPS

                window_bottom = np.random.randint(low=0, high=h)
                a1 = a1.iloc[window_bottom : window_bottom + self.ROLLOUT_TIMESTEPS]
                a2 = a2.iloc[window_bottom+1 : window_bottom + self.ROLLOUT_TIMESTEPS + 1]
                inds = range(window_bottom, window_bottom + self.ROLLOUT_TIMESTEPS + 1, 1)

                a1.reset_index(inplace=True, drop=True)
                a2.reset_index(inplace=True, drop=True)

                a1_col = a1[cols]
                a2_col = a2[cols]

                result["obj_pos_X"], result["obj_vel_X"], result["tip_pos_X"], result["tip_vel_X"], result["tip_con_X"] = self.convert_state(a1_col)
                result["obj_pos_Y"], result["obj_vel_Y"], result["tip_pos_Y"], result["tip_vel_Y"], result["tip_con_Y"] = self.convert_state(a2_col)
                
                ret.extend([result])
                internal_inds[index] = np.array(inds)
            else:
                to_del.append(i)

        traj_inds = np.delete(traj_inds, to_del)

        return ret, traj_inds, internal_inds

    def get_batch_of_trajectories_alt(self, df1, df2, batch_size, trajectory_ids = None):
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

        ret = []
        indeces_used = []
        internal_inds = {}

        traj_inds = df1.trajectory.unique()
        if trajectory_ids is not None:
            traj_inds = trajectory_ids
        
        df_1 = df1.set_index("trajectory", drop=False)
        df_2 = df2.set_index("trajectory", drop=False)

        while(True):
            index = np.random.choice(traj_inds, size=1, replace=False)[0]
        
            result = {} 
            a1 = df_1.loc[[index]]    
            a2 = df_2.loc[[index]]
            if a1.shape[0] > self.ROLLOUT_TIMESTEPS:
                # Get only the window slice
                h = len(a1) - self.ROLLOUT_TIMESTEPS

                window_bottom = np.random.randint(low=0, high=h)
                a1 = a1.iloc[window_bottom : window_bottom + self.ROLLOUT_TIMESTEPS]
                a2 = a2.iloc[window_bottom+1 : window_bottom + self.ROLLOUT_TIMESTEPS + 1]
                inds = range(window_bottom, window_bottom + self.ROLLOUT_TIMESTEPS + 1, 1)

                a1.reset_index(inplace=True, drop=True)
                a2.reset_index(inplace=True, drop=True)

                a1_col = a1[cols]
                a2_col = a2[cols]

                result["obj_pos_X"], result["obj_vel_X"], result["tip_pos_X"], result["tip_vel_X"], result["tip_con_X"] = self.convert_state(a1_col)
                result["obj_pos_Y"], result["obj_vel_Y"], result["tip_pos_Y"], result["tip_vel_Y"], result["tip_con_Y"] = self.convert_state(a2_col)
                
                indeces_used.append(index)
                ret.extend([result])
                internal_inds[index] = np.array(inds)

            if len(ret) >= batch_size:
                break

        return ret, indeces_used, internal_inds


    def get_trajectories(self, df_1, df_2, batch_size, graph_creator, trajectory_ids = None):

        b_trajs, traj_inds, internal_inds = self.get_batch_of_trajectories_alt(df_1, df_2, batch_size, trajectory_ids = trajectory_ids)
        
        traj_len = self.ROLLOUT_TIMESTEPS
        X_g = np.empty((0,traj_len))
        Y_g = np.empty((0,traj_len))
        i_np = np.empty((0, traj_len+1))
        
        for b_traj, ind in zip(b_trajs, traj_inds):
            Xs = []
            Ys = []
            Is = []
            
            for obj_pos_x, obj_vel_x, tip_pos_x, tip_vel_x, tip_con_x, obj_pos_y, obj_vel_y, tip_pos_y, tip_vel_y, tip_con_y in zip(b_traj['obj_pos_X'],b_traj["obj_vel_X"],b_traj['tip_pos_X'],b_traj['tip_vel_X'],b_traj['tip_con_X'],
                                                                                                              b_traj['obj_pos_Y'],b_traj["obj_vel_Y"],b_traj['tip_pos_Y'],b_traj['tip_vel_Y'],b_traj['tip_con_Y']):
                X_dict = graph_creator(obj_pos_x, obj_vel_x, tip_pos_x, tip_vel_x, tip_con_x)
                Y_dict = graph_creator(obj_pos_y, obj_vel_y, tip_pos_y, tip_vel_y, tip_con_y)
                Xs.extend([X_dict])
                Ys.extend([Y_dict])
                

            Is = internal_inds[ind]
            Xs = np.array([Xs])
            Ys = np.array([Ys])
            X_g = np.concatenate((X_g, Xs), axis=0)
            Y_g = np.concatenate((Y_g, Ys), axis=0)
            i_np = np.concatenate((i_np, [Is]), axis=0)
        
        X_g = X_g.T
        Y_g = Y_g.T
        i_np = i_np.T
        
        return X_g, Y_g, i_np


    def convert_trajectories(self, X_g,Y_g):
        traj_X = []
        traj_Y = []
        for i in range(self.ROLLOUT_TIMESTEPS):

            X = utils_tf.data_dicts_to_graphs_tuple(X_g[i])
            Y = utils_tf.data_dicts_to_graphs_tuple(Y_g[i])

            traj_X.extend([X])
            traj_Y.extend([Y])
        return traj_X, traj_Y


    def prediction_to_next_state_velocity(self, input_graph, predicted_graph, target_graph):
        is_fixed_to_move = input_graph.nodes[..., 4:5]

        new_vel = is_fixed_to_move * target_graph.nodes[..., 2:4] + (1-is_fixed_to_move) * predicted_graph.nodes
        new_pos = is_fixed_to_move * target_graph.nodes[..., :2] + (1-is_fixed_to_move) * (input_graph.nodes[..., :2] + predicted_graph.nodes * self.STEP_SIZE)
        
        new_nodes = tf.concat([new_pos, new_vel, input_graph.nodes[..., 4:5]], axis=-1)

        input_graph = input_graph.replace(nodes=new_nodes)
        input_graph = input_graph.replace(globals=target_graph.globals)
        
        return input_graph


    def prediction_to_next_state_position(self, input_graph, predicted_graph, target_graph):
        is_fixed_to_move = input_graph.nodes[..., 4:5]

        new_vel = target_graph.nodes[..., 2:4]
        new_pos = is_fixed_to_move * target_graph.nodes[..., :2] + (1-is_fixed_to_move) * predicted_graph.nodes
        
        new_nodes = tf.concat([new_pos, new_vel, input_graph.nodes[..., 4:5]], axis=-1)

        input_graph = input_graph.replace(nodes=new_nodes)
        input_graph = input_graph.replace(globals=target_graph.globals)

        return input_graph


    def my_tf_round(self, x, decimals = 0):
        multiplier = tf.constant(10**decimals, dtype=x.dtype)
        return tf.round(x * multiplier) / multiplier


    def predict_trajectory_position(self, model, Xs, Ys, silent=True):

        pred_states = [] 
        real_states = []
        # Initialise the input state
        input_state = Xs[0]
        
        for i,X,Y in zip(range(self.ROLLOUT_TIMESTEPS),Xs,Ys): 
            predictions = model(input_state, 1)
            next_state = self.prediction_to_next_state_position(input_state, predictions[0], Y)
            input_state = next_state
            pred_states.extend([next_state])
            real_states.extend([Y])

            if not silent:
                print("Input")
                print(self.my_tf_round(input_state.nodes,4))
                print("Predictions")
                print(self.my_tf_round(predictions[0].nodes,4))
                print("-----------------")
                print("Next state")
                print(self.my_tf_round(next_state.nodes,4))
                print("Target")
                print(self.my_tf_round(Y.nodes,4))
                print("-----------------")
                print("-----------------")

        return pred_states, real_states
            

    def predict_trajectory_velocity(self, model, Xs, Ys, silent=True):

        pred_states = [] 
        real_states = []
        # Initialise the input state
        input_state = Xs[0]

        for i,X,Y in zip(range(self.ROLLOUT_TIMESTEPS),Xs,Ys):
            if model is not None:
                predictions = model(input_state, 1)
                next_state = self.prediction_to_next_state_velocity(input_state, predictions[0], Y)
                input_state = next_state
                pred_states.extend([next_state])
            real_states.extend([Y])

            if not silent:
                print("Input")
                print(self.my_tf_round(input_state.nodes,4))
                print("Predictions")
                print(self.my_tf_round(predictions[0].nodes,4))
                print("-----------------")
                print("Next state")
                print(self.my_tf_round(next_state.nodes,4))
                print("Target")
                print(self.my_tf_round(Y.nodes,4))
                print("-----------------")
                print("-----------------")

        return pred_states, real_states
            
        
        