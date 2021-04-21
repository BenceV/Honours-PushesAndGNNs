import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon

def velocity_loss(target_ls, prediction_ls):
    loss = np.mean(np.sum((target_ls-prediction_ls)**2, axis=1), axis=0)
    return loss

def order_vertices(ray):
    """
    This function reorders the vertices in an array for polygon vertices.
    We do this by:
        1. Compute the centroid of the "polygon"
        2. Compute the rays from the centroid to each of your "vertices".
        3. Use atan2 to compute the angle of this ray from the horizontal
        4. Sort the vertices according to this angle.
        ------------
        Source: https://uk.mathworks.com/matlabcentral/answers/366317-how-to-correctly-order-polygon-vertices , 
                by Matt J

    Parameters
    ----------
    ray : np.array ,
        A numpy array of 2D points.

    Raises
    ------
    ValueError, 
        If ray is not a ndarray of 2D points 

    Returns
    -------
    sorted_ray : np.array ,
        Numpy array but sorted.

    """
    if (not len(ray.shape) == 2) or (not ray.shape[1] == 2):
        raise ValueError()
    try:
        center = np.mean(ray,axis=0)
        vectors = ray - center
        angles = np.arctan2(vectors[:,1],vectors[:,0])
        arr1inds = angles.argsort()
        sorted_ray = ray[arr1inds[::-1]]
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print()
    return sorted_ray


def visualise_trajectory(pred_trajectory, real_trajectory, number_of_steps=None):
    fig, ax = plt.subplots()

    number_of_steps = len(pred_trajectory)
    # ------------------Extraction------------------
    
    pred_state_0 = pred_trajectory[0]
    pred_nodes_pos_0 = pred_state_0.nodes[:-1,:2].numpy() 
    pred_center_pos_0 = pred_nodes_pos_0[-1]
    pred_corners_pos_0 = pred_nodes_pos_0[:-1]
    
    real_state_0 = real_trajectory[0]
    real_nodes_pos_0 = real_state_0.nodes[:-1,:2].numpy() 
    real_center_pos_0 = real_nodes_pos_0[-1]
    real_corners_pos_0 = real_nodes_pos_0[:-1]
    
    real_tip_pos_0 = real_state_0.nodes[-1,:2].numpy() 
    

    center_loc_x = np.mean(real_nodes_pos_0[:, 0])
    center_loc_y = np.mean(real_nodes_pos_0[:, 1])

    zoom_out = 0.2

    ax.axis('equal')
    ax.set_xlim(center_loc_x-zoom_out,center_loc_x+zoom_out)
    ax.set_ylim(center_loc_y-zoom_out,center_loc_y+zoom_out)


    #Rectangle
    rect_pred = Polygon(order_vertices(pred_corners_pos_0), closed=False, animated=False, alpha=0.2, color='b')
    rect_real = Polygon(order_vertices(real_corners_pos_0), closed=False, animated=False, alpha=0.2, color='r')
    ax.add_patch(rect_real)
    ax.add_patch(rect_pred)

    nodes_pred, = ax.plot([], [], 'bo',label="Predicted center")
    nodes_real, = ax.plot([], [], 'ro',label="Real center")
    
    
    #end_effector_pred, = ax.plot([], [], 'bx',label="Predicted End-Effector Tip")
    end_effector_real, = ax.plot([], [], 'kx',label="End-Effector Tip")
    
    values_at_i = ax.text(center_loc_x - zoom_out*0.95, center_loc_y - zoom_out*0.7, "Error: " + str(), size=10,
            va="baseline", ha="left", multialignment="left",
            bbox=dict(fc="none"))


    fig.suptitle("Visualisation of trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc=2)

    def init():
        nodes_pred.set_data([], [])
        nodes_real.set_data([], [])
        
        end_effector_real.set_data([], [])
        
        rect_real.set_xy(order_vertices(real_corners_pos_0))
        rect_pred.set_xy(order_vertices(pred_corners_pos_0))
        
        values_at_i.set_text("Error: " + str(0.0))
        
        return (nodes_pred, nodes_real,
                end_effector_real,
                rect_pred, rect_real,
                values_at_i)

    def animate(i):
        pred_state = pred_trajectory[i]
        pred_nodes_pos_i = pred_state.nodes[:-1,:2].numpy() 
        pred_center_pos_i = pred_nodes_pos_i[-1]
        pred_corners_pos_i = pred_nodes_pos_i[:-1]
        
        real_state = real_trajectory[i]
        real_nodes_pos_i = real_state.nodes[:-1,:2].numpy() 
        real_center_pos_i = real_nodes_pos_i[-1]
        real_corners_pos_i = real_nodes_pos_i[:-1]
        
        
        real_tip_pos_i = real_state.nodes[-1,:2].numpy() 

        # Centers: Real
        x_cors = real_center_pos_i[0]
        y_cors = real_center_pos_i[1]    
        nodes_real.set_data(x_cors, y_cors)
        
        # Centers: Pred
        x_cors = pred_center_pos_i[0]
        y_cors = pred_center_pos_i[1]    
        nodes_pred.set_data(x_cors, y_cors)

        # End-effector Tip: Position: Real
        x_tip = real_tip_pos_i[0]
        y_tip = real_tip_pos_i[1]
        end_effector_real.set_data(x_tip, y_tip)
        
        # Rectangle: Real
        nods = np.copy(real_corners_pos_i)
        nods = order_vertices(nods)
        rect_real.set_xy(nods)
        
        # Rectangle: Pred
        nods = np.copy(pred_corners_pos_i)
        nods = order_vertices(nods)
        rect_pred.set_xy(nods)

        # Update title
        ax.set_title("Current timestep: "+str(i))
        error = velocity_loss(real_nodes_pos_i, pred_nodes_pos_i)
        values_at_i.set_text("Error: {:.6f}".format(error))

        return (nodes_pred, nodes_real,
                end_effector_real,
                rect_pred, rect_real,
                values_at_i)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=number_of_steps, interval=1, blit=False, repeat=True)

    return fig, ax, anim


def visualise_groundtruth(real_trajectory, number_of_steps):
    fig, ax = plt.subplots()

    # ------------------Extraction------------------
    
    real_state_0 = real_trajectory[0]
    real_nodes_pos_0 = real_state_0.nodes[:-1,:2].numpy() 
    real_center_pos_0 = real_nodes_pos_0[-1]
    real_corners_pos_0 = real_nodes_pos_0[:-1]
    
    real_tip_pos_0 = real_state_0.nodes[-1,:2].numpy() 
    

    center_loc_x = np.mean(real_nodes_pos_0[:, 0])
    center_loc_y = np.mean(real_nodes_pos_0[:, 1])

    zoom_out = 0.2

    ax.axis('equal')
    ax.set_xlim(center_loc_x-zoom_out,center_loc_x+zoom_out)
    ax.set_ylim(center_loc_y-zoom_out,center_loc_y+zoom_out)


    #Rectangle
    rect_real = Polygon(order_vertices(real_corners_pos_0), closed=False, animated=False, alpha=0.2, color='r')
    ax.add_patch(rect_real)
    
    nodes_real, = ax.plot([], [], 'ro',label="Real center")
    
    
    #end_effector_pred, = ax.plot([], [], 'bx',label="Predicted End-Effector Tip")
    end_effector_real, = ax.plot([], [], 'kx',label="End-Effector Tip")
    
    values_at_i = ax.text(center_loc_x - zoom_out*0.95, center_loc_y - zoom_out*0.7, "", size=10,
            va="baseline", ha="left", multialignment="left",
            bbox=dict(fc="none"))


    fig.suptitle("Visualisation of trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc=2)

    def init():
        nodes_real.set_data([], [])
        
        end_effector_real.set_data([], [])
        
        rect_real.set_xy(order_vertices(real_corners_pos_0))
        
        return (nodes_real,
                end_effector_real, rect_real)

    def animate(i):
        real_state = real_trajectory[i]
        real_nodes_pos_i = real_state.nodes[:-1,:2].numpy() 
        real_center_pos_i = real_nodes_pos_i[-1]
        real_corners_pos_i = real_nodes_pos_i[:-1]
        
        
        real_tip_pos_i = real_state.nodes[-1,:2].numpy() 

        # Centers: Real
        x_cors = real_center_pos_i[0]
        y_cors = real_center_pos_i[1]    
        nodes_real.set_data(x_cors, y_cors)
        
        # End-effector Tip: Position: Real
        x_tip = real_tip_pos_i[0]
        y_tip = real_tip_pos_i[1]
        end_effector_real.set_data(x_tip, y_tip)
        
        # Rectangle: Real
        nods = np.copy(real_corners_pos_i)
        nods = order_vertices(nods)
        rect_real.set_xy(nods)

        # Update title
        ax.set_title("Current timestep: "+str(i))

        return (nodes_real,
                end_effector_real, rect_real)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=number_of_steps, interval=1, blit=False, repeat=True)

    return fig, ax, anim




def visualise_one_step(pred_trajectory, real_trajectory):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # ------------------Extraction------------------
    # Timestep: 0
    pred_state_0 = pred_trajectory[0]
    pred_nodes_pos_0 = pred_state_0.nodes[:-1,:2].numpy() 
    pred_center_pos_0 = pred_nodes_pos_0[-1]
    pred_corners_pos_0 = pred_nodes_pos_0[:-1]
    
    real_state_0 = real_trajectory[0]
    real_nodes_pos_0 = real_state_0.nodes[:-1,:2].numpy() 
    real_center_pos_0 = real_nodes_pos_0[-1]
    real_corners_pos_0 = real_nodes_pos_0[:-1]
    
    tip_pos_0 = real_state_0.nodes[-1,:2].numpy() 
    
    # Timestep: 1
    pred_state_1 = pred_trajectory[1]
    pred_nodes_pos_1 = pred_state_1.nodes[:-1,:2].numpy() 
    pred_center_pos_1 = pred_nodes_pos_1[-1]
    pred_corners_pos_1 = pred_nodes_pos_0[:-1]
    
    real_state_1 = real_trajectory[1]
    real_nodes_pos_1 = real_state_1.nodes[:-1,:2].numpy() 
    real_center_pos_1 = real_nodes_pos_1[-1]
    real_corners_pos_1 = real_nodes_pos_1[:-1]
    
    tip_pos_1 = real_state_1.nodes[-1,:2].numpy() 
    
    
    
    
    
    # ------------------Axes------------------
    zoom_out = 0.5

    center_loc_x = np.mean(real_corners_pos_0[:, 0])
    center_loc_y = np.mean(real_corners_pos_0[:, 1])
    
    ax1.axis('equal')
    ax1.set_xlim(center_loc_x-zoom_out,center_loc_x+zoom_out)
    ax1.set_ylim(center_loc_y-zoom_out,center_loc_y+zoom_out)

    center_loc_x = np.mean(real_corners_pos_1[:, 0])
    center_loc_y = np.mean(real_corners_pos_1[:, 1])
    
    ax2.axis('equal')
    ax2.set_xlim(center_loc_x-zoom_out,center_loc_x+zoom_out)
    ax2.set_ylim(center_loc_y-zoom_out,center_loc_y+zoom_out)

    # ------------------Rectangle------------------
    rect_real = Polygon(order_vertices(real_corners_pos_0), closed=False, animated=False, alpha=0.2, color='r')
    rect_pred = Polygon(order_vertices(pred_corners_pos_0), closed=False, animated=False, alpha=0.2, color='b')
    ax1.add_patch(rect_real)
    ax1.add_patch(rect_pred)
    
    rect_real = Polygon(order_vertices(real_corners_pos_1), closed=False, animated=False, alpha=0.2, color='r')
    rect_pred = Polygon(order_vertices(pred_corners_pos_1), closed=False, animated=False, alpha=0.2, color='b')
    ax2.add_patch(rect_real)
    ax2.add_patch(rect_pred)
    
    x_tip = tip_pos_0[0]
    y_tip = tip_pos_0[1]
    end_effector_1, = ax1.plot([], [], 'kx',label="End-Effector Tip")
    end_effector_1.set_data(x_tip, y_tip)
    x_tip = tip_pos_1[0]
    y_tip = tip_pos_1[1]
    end_effector_2, = ax2.plot([], [], 'kx',label="End-Effector Tip")
    end_effector_2.set_data(x_tip, y_tip)
    
    values_at_i = ax2.text(center_loc_x - zoom_out*0.95, center_loc_y - zoom_out*0.7, "", size=10,
            va="baseline", ha="left", multialignment="left",
            bbox=dict(fc="none"))


    fig.suptitle("Visualisation of trajectory")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax1.legend(loc=2)
    ax2.legend(loc=2)

   
    return fig, ax1, ax2

