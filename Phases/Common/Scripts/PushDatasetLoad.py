
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
from sklearn.model_selection import train_test_split

#Import graph nets
from graph_nets import blocks
from graph_nets import utils_tf
from graph_nets import utils_np


def load_dataset(path):
    head, tail = os.path.split(path)
    step_size = (int(tail.split("_")[1][4:-2]) / 1000.0)
    dataset_1 = pd.read_csv(path)
    dataset_1["vel_acc"]= dataset_1["base_vel"].map(str) + '-' + dataset_1["base_acc"].map(str)
    return dataset_1, step_size
    
def collect_trajectory_indeces(dataframe, ts = 0.001, vel_accs = None):

    # Retrieve only the requested vel_acc pairs
    if vel_accs is not None:
        vals = dataframe.loc[dataframe['vel_acc'].isin(vel_accs)]
        dataframe = vals.copy()

    
    # Create describtion column
    df_vel_acc = dataframe.groupby(['vel_acc']).size().reset_index().rename(columns={0:'count'})
    np_vel_acc = df_vel_acc[['vel_acc']].to_numpy()
    
    dfs = {}
    for vel_acc in np_vel_acc:
        vel_acc = vel_acc.squeeze()
        df_vel_acc = dataframe.loc[dataframe['vel_acc'] == vel_acc]
        dfs[str(vel_acc)] = df_vel_acc
    
    pos_list = ["o_t_r_x", 
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
                "e_pos_y"]
    
    vel_list = ["o_t_r_x_v", 
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
                "e_vel_y"]
    
    # Create velocity columns
    dataframe[vel_list] = dataframe[pos_list].diff(periods=-1)*-1 / ts
    
    ids = dataframe.groupby(['trajectory']).last()['id'].to_numpy()
    dataframe.loc[ids, vel_list] =  0.0
    
    inds={}
    tups={}
    for key in dfs.keys():
        inds[key] = dfs[key].trajectory.unique()
        pds = dfs[key].groupby(['trajectory'])['trajectory'].count()
        tups[key] = pds
    
    return inds, dataframe

# Split the collected inds for each type of operation according
def create_train_valid_test(indeces_dict, test_size):
    tr_inds = np.array([])
    va_inds = np.array([])
    te_inds = np.array([])
    for key in indeces_dict.keys():
        tr, va_te = train_test_split(indeces_dict[key], test_size = test_size)
        va, te = train_test_split(va_te, test_size = 0.5)
        tr_inds = np.append(tr_inds,tr)
        va_inds = np.append(va_inds,va)
        te_inds = np.append(te_inds,te)
        
    return tr_inds, va_inds, te_inds

def collect_states(indeces, df):
    df2 = df.set_index("trajectory", drop=False)
    a = df2.loc[indeces]
    a.reset_index(inplace=True, drop=True)
    return a 

def remove_effector(output_op, target_nodes):
    n_nodes_in_train = output_op.shape[0]
    boolean_l = np.array([True]*n_nodes_in_train)
    indices_of_fixed = np.array(list(range(5,n_nodes_in_train,6)))
    boolean_l[indices_of_fixed] = False

    changed_output_op = tf.boolean_mask(output_op,boolean_l)
    changed_target_nodes = tf.boolean_mask(target_nodes,boolean_l)
    return changed_output_op, changed_target_nodes


def outlier_remover(df1, df2, zero_thr):
    vel_acc_list = list(df1['vel_acc'].unique())
    vel_list_2 = [
        "o_t_r_x_v", 
        "o_t_r_y_v",
        "o_t_l_x_v",
        "o_t_l_y_v",
        "o_b_r_x_v",
        "o_b_r_y_v",
        "o_b_l_x_v",
        "o_b_l_y_v"]

    df_1s = []
    df_2s = []
    
    for vel_acc in vel_acc_list:
        df_v_a_1 = df1.loc[df1['vel_acc'] == vel_acc]       
        df_v_a_2 = df2.loc[df2['vel_acc'] == vel_acc]
        v_a_np_1 = df_v_a_1[vel_list_2].to_numpy()
        v_a_np_1 = np.abs(v_a_np_1)
        
        
        # Get the threshold
        node_mean_speed = np.mean(v_a_np_1, axis=0)
        node_std_speed = np.std(v_a_np_1, axis=0)
        thr_h = node_mean_speed + node_std_speed * 3
        thr_l = node_mean_speed - node_std_speed * 3
        
        # Find outlier rows in the numpy matrix of X
        sad_1_h = v_a_np_1 > thr_h 
        sad_1_l = v_a_np_1 < thr_l
        
        # Find close to 0 velocities
        
        v_a_np_1_s = v_a_np_1.reshape((len(v_a_np_1), 2, 4))
        v_a_np_1_s_s = np.sum(np.power(v_a_np_1_s,2),axis=1)
        v_avg_np_1 = np.mean(v_a_np_1_s_s, axis=1)
        sad_1_0 = v_avg_np_1 < zero_thr

        sad_row_1_h = np.bitwise_or.reduce(sad_1_h, axis=1)
        sad_row_1_l = np.bitwise_or.reduce(sad_1_l, axis=1)
        sad_row_1 = np.logical_or(sad_row_1_h, sad_row_1_l)
        sad_row_1 = np.logical_or(sad_row_1, sad_1_0)
        # Select rows that are not outliers in both X and Y
        df_v_a_1 = df_v_a_1[~sad_row_1]
        df_v_a_2 = df_v_a_2[~sad_row_1]
        
        # Find outlier rows in the numpy matrix of Y
        v_a_np_2 = df_v_a_2[vel_list_2].to_numpy()
        v_a_np_2 = np.abs(v_a_np_2)

        sad_2_h = v_a_np_2 > thr_h 
        sad_2_l = v_a_np_2 < thr_l
        
        # Find close to 0 velocities
        v_a_np_2_s = v_a_np_2.reshape((len(v_a_np_2), 2, 4))
        v_a_np_2_s_s = np.sum(np.power(v_a_np_2_s,2),axis=1)
        v_avg_np_2 = np.min(v_a_np_2_s_s, axis=1)
        sad_2_0 = v_avg_np_2 < zero_thr

        sad_row_2_h = np.bitwise_or.reduce(sad_2_h, axis=1)
        sad_row_2_l = np.bitwise_or.reduce(sad_2_l, axis=1)
        sad_row_2 = np.logical_or(sad_row_2_h, sad_row_2_l)
        sad_row_2 = np.logical_or(sad_row_2, sad_2_0)
        
        # Select rows that are not outliers in both X and Y
        df_v_a_1 = df_v_a_1[~sad_row_2]
        df_v_a_2 = df_v_a_2[~sad_row_2]
        
        df_1s.append(df_v_a_1)
        df_2s.append(df_v_a_2)
        
    df_1 = pd.concat(df_1s)
    df_2 = pd.concat(df_2s)

    df_1.reset_index(drop=True, inplace=True)
    df_2.reset_index(drop=True, inplace=True)
    #df_1 = df_1.reindex(index=range(df_1.shape[0]))
    #df_2 = df_2.reindex(index=range(df_2.shape[0]))
    
    return df_1, df_2


def dataset_formating(df_1, df_2):
    df_1a = df_1.copy()
    df_2a = df_2.copy()
    
    saa = df_1a.groupby(["trajectory"])[["trajectory"]].count()
    
    saa["traj"] = saa.index.values.tolist()
    cols = saa.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    saa = saa[cols]
    saa_np = saa.to_numpy()
    av_s = np.mean(saa_np,axis=0)[1]
    max_rollout_length = np.max(saa_np[saa_np[:,1] > av_s/4.0],axis=0)[1]
    max_batch_size = len(saa_np[saa_np[:,1] > av_s/4.0])
    
    traj_inds = saa_np[saa_np[:,1] > av_s/4.0][:,0]
    
    df_1a.set_index('trajectory', drop=False, inplace = True)
    df_2a.set_index('trajectory', drop=False, inplace = True)
    
    df_1a = df_1a.loc[traj_inds]
    df_2a = df_2a.loc[traj_inds]
    
    df_1a.reset_index(drop=True, inplace = True)
    df_2a.reset_index(drop=True, inplace = True)
    
    return max_rollout_length, max_batch_size, df_1a, df_2a