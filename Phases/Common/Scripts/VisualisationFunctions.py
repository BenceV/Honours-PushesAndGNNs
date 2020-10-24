import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

def plot_groundtruth(true_trajectory_te_np,
                    spring_const_te, damping_ratio_te, 
                    contact_nodes_te, 
                    path_animations, 
                    file_name,
                    num_time_steps=50, step_size = 0.1):
    # Visualize trajectories
    fig_animate3 = plt.figure(1, figsize=(6, 6))
    ax3 = fig_animate3.add_subplot(1, 1, 1)

    # Visualize trajectories for number_of_masses = 4
    edges = []
    contact_edges = []
    time_text = ax3.text(-1.8, 1.8, "Frame: "+str(0),fontsize=12)
    num_nodes = true_trajectory_te_np.shape[1]

    def init():
        for i in range(num_nodes-1):
            if(i==0):
                edges.append(ax3.plot([],[], linestyle='-', marker='o', markersize=5, color="red",label="Truth"))
            else:
                edges.append(ax3.plot([],[], linestyle='-', marker='o', markersize=5, color="red"))
        
        for i in range(len(contact_nodes_te)):
            contact_edges.append(ax3.plot([],[], linestyle='-', marker='o', markersize=5, color="darkred"))
            
        return edges, contact_edges, time_text
        
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)


    def animate(z):
        true_tr = true_trajectory_te_np[z]                               
        time_text.set_text("Frame: "+str(z))
        for i in range(num_nodes-2):
            left = i
            right = i+1
            
            if (right==num_nodes):
                break
                        
            edges[i][0].set_data([true_tr[left,0],true_tr[right,0]],[true_tr[left,1],true_tr[right,1]])
        edges[3][0].set_data([true_tr[0,0],true_tr[3,0]],[true_tr[0,1],true_tr[3,1]])
        
        for node_n,i in zip(contact_nodes_te,range(len(contact_nodes_te))):
            contact_edges[i][0].set_data([true_tr[node_n,0],true_tr[4,0]],[true_tr[node_n,1],true_tr[4,1]])
            
        legend = ax3.legend()
        return edges, contact_edges, legend,time_text

    ax3.set_title("Trajectory: test, k = "+str(np.round(spring_const_te,2))+", d = "+str(np.round(damping_ratio_te,2)))
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")

    anim = animation.FuncAnimation(fig_animate3, animate,init_func = init, interval = step_size * 1000, frames=num_time_steps, blit=False, repeat=False)
    #Save animation
    if os.path.exists(os.path.join(path_animations,"GroundTruth",file_name+".gif")):
        print("The file: "+ file_name+".gif" + "already exists. Delete it before saving a new animation!")
    else:
        if not os.path.exists(os.path.join(path_animations,"GroundTruth")):
            os.mkdir(os.path.join(path_animations,"GroundTruth"))
                    
        anim.save(os.path.join(path_animations,"GroundTruth",file_name+".gif"), writer='imagemagick', fps=20)

    return fig_animate3,ax3,anim

    

def plot_groundtruth_and_prediction(true_trajectory_te_np,
                                    predicted_trajectory_te_np, 
                                    spring_const_te, damping_ratio_te, 
                                    contact_nodes_te, 
                                    path_animations, 
                                    file_name,
                                    num_time_steps=50, step_size = 0.1):
    
    fig_animate3 = plt.figure(1, figsize=(6, 6))
    ax3 = fig_animate3.add_subplot(1, 1, 1)

    # Visualize trajectories for number_of_masses = 4

    edges = []
    edges1 = []
    contact_edges = []
    contact_edges1 = []

    time_text = ax3.text(-1.8, 1.8, "Frame: "+str(0),fontsize=12)
    num_nodes = true_trajectory_te_np.shape[1]
        
    def init():
        for i in range(num_nodes-1):
            if(i==0):
                edges.append(ax3.plot([],[], linestyle='-', marker='o', markersize=5, color="red",label="Truth"))
                edges1.append(ax3.plot([],[], linestyle='-', marker='o', markersize=5, color="blue",label="Prediction"))
            else:
                edges.append(ax3.plot([],[], linestyle='-', marker='o', markersize=5, color="red"))
                edges1.append(ax3.plot([],[], linestyle='-', marker='o', markersize=5, color="blue"))
        
        for i in range(len(contact_nodes_te)):
            contact_edges.append(ax3.plot([],[], linestyle='-', marker='o', markersize=5, color="darkred"))   
            contact_edges1.append(ax3.plot([],[], linestyle='-', marker='o', markersize=5, color="navy"))
            
        return edges,edges1,contact_edges,contact_edges1,time_text
        
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)


    def animate(z):
        
        true_tr = true_trajectory_te_np[z]
        pred_tr = predicted_trajectory_te_np[z]                              
        time_text.set_text("Frame: "+str(z))
        for i in range(num_nodes-2):
            left = i
            right = i+1
            
            if (right==num_nodes):
                break
                        
            edges[i][0].set_data([true_tr[left,0],true_tr[right,0]],[true_tr[left,1],true_tr[right,1]])
            edges1[i][0].set_data([pred_tr[left,0],pred_tr[right,0]],[pred_tr[left,1],pred_tr[right,1]])
        edges[3][0].set_data([true_tr[0,0],true_tr[3,0]],[true_tr[0,1],true_tr[3,1]])   
        edges1[3][0].set_data([pred_tr[0,0],pred_tr[3,0]],[pred_tr[0,1],pred_tr[3,1]])
        
        for node_n,i in zip(contact_nodes_te,range(len(contact_nodes_te))):
            contact_edges[i][0].set_data([true_tr[node_n,0],true_tr[4,0]],[true_tr[node_n,1],true_tr[4,1]])
            contact_edges1[i][0].set_data([pred_tr[node_n,0],pred_tr[4,0]],[pred_tr[node_n,1],pred_tr[4,1]])
        
        legend = ax3.legend()
        return legend,edges,edges1,contact_edges,contact_edges1,time_text


    ax3.set_title("Rollout Trajectory: test, k = "+str(np.round(spring_const_te,2))+", d = "+str(np.round(damping_ratio_te,2)))
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")

    anim = animation.FuncAnimation(fig_animate3, animate,init_func = init, interval = step_size * 1000, frames=num_time_steps, blit=False, repeat=False)
    #Save animation
    if os.path.exists(os.path.join(path_animations,"FullRollout",file_name+".gif")):
        print("The file: "+ file_name+".gif" + "already exists. Delete it before saving a new animation!")
    else:
        if not os.path.exists(os.path.join(path_animations,"FullRollout")):
            os.mkdir(os.path.join(path_animations,"FullRollout"))
                    
        anim.save(os.path.join(path_animations,"FullRollout",file_name+".gif"), writer='imagemagick', fps=20)

                
    return fig_animate3,ax3,anim