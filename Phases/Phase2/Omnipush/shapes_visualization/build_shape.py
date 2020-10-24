import dxfgrabber
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patch
import copy
import time, pdb
from shapely import geometry
import matplotlib.ticker as ticker
import os

global data_directory

def get_dxf(dxf, basic_rot, rotation = 0, do_plot = False, fig = None, ax = None, num_circle = 0, x_pos = 0, y_pos = 0, color = 'k', weight_value = 'a'):
    
    data = {}
    data['x'] = []
    data['y'] = []
    for entity in dxf.entities:
        if entity.dxftype == 'LINE':
            # Extract data
            x_line = np.array([entity.start[0], entity.end[0]])
            y_line = np.array([entity.start[1], entity.end[1]])
            # Rotation
            rotation_rad = rotation*np.pi/180
            aux_x_line = copy.deepcopy(x_line)
            x_line = x_line*np.cos(rotation_rad) - y_line*np.sin(rotation_rad) +x_pos
            y_line = aux_x_line*np.sin(rotation_rad) + y_line*np.cos(rotation_rad) + y_pos
            # Save data
            data['x'] += x_line.tolist()
            data['y'] += y_line.tolist()
            if do_plot:
                plt.plot(x_line, y_line, color=color)
            
        if entity.dxftype == 'CIRCLE' and num_circle > -1:
            rot_center = [0, 52.426]
            if num_circle:
                num_circle = 0
                rot_center = [0, 62.426]
            else:
                num_circle = -1
            aux_rot_center = copy.deepcopy(rot_center)
            rotation_rad = basic_rot*np.pi/180
            rot_center[0] = rot_center[0]*np.cos(rotation_rad) - rot_center[1]*np.sin(rotation_rad) +x_pos
            rot_center[1] = aux_rot_center[0]*np.sin(rotation_rad) + rot_center[1]*np.cos(rotation_rad) + y_pos
            if weight_value.lower() == 'b':
                ax.add_patch(plt.Circle(rot_center,entity.radius, color='r'))
            elif weight_value.lower() == 'c':
                ax.add_patch(plt.Circle(rot_center,entity.radius, color='k')) #big weight
        if entity.dxftype == 'ARC':
            
            ## Positive coordinate            
            rot_center = np.array(entity.center)
            aux_rot_center = copy.deepcopy(rot_center)
            rotation_rad = rotation*np.pi/180
            rot_center[0] = rot_center[0]*np.cos(rotation_rad) - rot_center[1]*np.sin(rotation_rad) + x_pos
            rot_center[1] = aux_rot_center[0]*np.sin(rotation_rad) + rot_center[1]*np.cos(rotation_rad) + y_pos
            
            # Save data
            alpha = (rotation + np.linspace(entity.start_angle, entity.end_angle, 100))*np.pi/180
            x_angle = rot_center[0] + entity.radius*(np.cos(alpha))+x_pos
            y_angle = rot_center[1] + entity.radius*(np.sin(alpha))+y_pos
            data['x'] += x_angle.tolist()
            data['y'] += y_angle.tolist()
            
            if do_plot:
                arc = patch.Arc(rot_center,entity.radius*2,entity.radius*2,rotation, entity.start_angle, entity.end_angle, color=color)
                ax.add_patch(arc)
            
            ## Negative coordinate
            rot_center = np.array(entity.center)
            rot_center[0] *=-1
            aux_rot_center = copy.deepcopy(rot_center)
            rotation_rad = rotation*np.pi/180
            rot_center[0] = rot_center[0]*np.cos(rotation_rad) - rot_center[1]*np.sin(rotation_rad) + x_pos
            rot_center[1] = aux_rot_center[0]*np.sin(rotation_rad) + rot_center[1]*np.cos(rotation_rad) + y_pos
            offset = 180
            
            # Save data
            alpha = (rotation + offset + np.linspace(-entity.start_angle, -entity.end_angle, 100))*np.pi/180
            x_angle = rot_center[0] + entity.radius*(np.cos(alpha))
            y_angle = rot_center[1] + entity.radius*(np.sin(alpha))
            data['x'] += x_angle.tolist()
            data['y'] += y_angle.tolist()
            
            if do_plot:
                ax.add_patch(patch.Arc(rot_center,entity.radius*2,entity.radius*2,offset+rotation, -entity.end_angle, -entity.start_angle, color=color))
    return data

def get_full_shape(shapename, shape_rotation = 0, x_pos = 0, y_pos = 0, tip_pos = None, do_plot = False, save_path = None, color = 'k'):
    data = {}
    data['x'] = []
    data['y'] = []

    if do_plot:
        plt.close()
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        # Plot tip
        if tip_pos is not None:
            tip_radius = 9.05/2
            ax.add_patch(plt.Circle(tip_pos,tip_radius, color=color))
    else:
        fig = None
        ax = None

    # Get data
    for i in range(4):
        side = int(shapename[2*i])
        num_circle = shapename[2*i+1].isupper()
        weight_value = shapename[2*i+1]
        dxf=dxfgrabber.readfile(data_directory + "/Part_{}.dxf".format(side))
        aux_data = get_dxf(dxf,shape_rotation+(i-1)*90, shape_rotation+(i-side)*90, do_plot, fig, ax, num_circle = num_circle, x_pos = x_pos, y_pos = y_pos, color = color, weight_value = shapename[2*i+1])
        data['x'] += aux_data['x']
        data['y'] += aux_data['y']
    if do_plot: 
        # Change scale
        scale_x = 1e3
        scale_y = 1e3
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
        plt.gca().xaxis.set_major_formatter(ticks_x)

        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
        plt.gca().yaxis.set_major_formatter(ticks_y)
        
        #plt.axis("off")
        plt.title(shapename)
        if save_path is not None:
            plt.savefig(save_path+'shape.png')
            plt.savefig(save_path+'shape_{}.png'.format(shapename))
        plt.show()
    return data

def check_collision(data,push_pose, radius_tip,  do_plot = False):
    # List points
    pointList = []
    for i in range(len(data['x'])):
        pointList.append((data['x'][i], data['y'][i]))
    poly = geometry.Polygon(pointList)

    point = geometry.Point(push_pose)
    circle = point.buffer(radius_tip)
    print('Intersect? ', poly.intersects(circle))
    plt.axis("off")
    '''
    if do_plot:
        x,y = circle.exterior.xy
        plt.plot(x,y,'b')
        plt.show()
    '''
    
def visualize_shape_given_data(shapename, input_data, size_data, output_data = None, do_plot = True,  color = 'k'):
    shape_rotation = 0
    x_pos = 0
    y_pos = 0
    if size_data == 7:
        x_pos = input_data[0]
        y_pos = input_data[1]
        shape_rotation = input_data[2]
        tip_pos = [input_data[3], input_data[4]]
        if output_data is not None:
            x_pos += output_data[0]
            y_pos += output_data[1]
            shape_rotation += output_data[2]
            tip_pos = [tip_pos[0] + input_data[5], tip_pos[1] + input_data[6]]
    elif size_data == 5:
        shape_rotation = input_data[0]
        tip_pos = [input_data[1], input_data[2]]
        if output_data is not None:
            x_pos += output_data[0]
            y_pos += output_data[1]
            shape_rotation += output_data[2]
            tip_pos = [tip_pos[0] + input_data[3], tip_pos[1] + input_data[4]]
    else:
        tip_pos = [input_data[0], input_data[1]]
        if output_data is not None:
            x_pos += output_data[0]
            y_pos += output_data[1]
            shape_rotation += output_data[2]
            tip_pos[0] += 0.05*np.cos(input_data[2])
            tip_pos[1] += 0.05*np.sin(input_data[2])
    data = get_full_shape(shapename, shape_rotation*180/np.pi, x_pos*1000, y_pos*1000, tip_pos = [tip_pos[0]*1000, tip_pos[1]*1000], do_plot = do_plot, color = color) #has to be in mm
    return data 

if __name__=='__main__':

    shapename = '1b4a2a2a'
    shape_rotation = 0
    push_pose = [55,0]
    radius_tip = 5
    do_plot = True
    x_pos = -1000; y_pos= -20
    data_directory = os.path.dirname(os.path.realpath(__file__))
    ## Visualize shape
    data = get_full_shape(shapename, shape_rotation, x_pos, y_pos, tip_pos = [-1000, 40], do_plot = do_plot)
    check_collision(data, push_pose, radius_tip, do_plot)
    
    
    ## Examples using data from the dataset
    shapename = '1a1a3a3a'
    size_data = 3; input_data = [-0.07409951,  0.01117996, -0.75870787]; output_data =  [0.02644353, -0.01257217,  0.47728441]
    #size_data = 5; input_data = [ 1.9047198 ,  0.01372386, -0.07367079,  0.02060622,  0.04555638]; output_data = [ 3.21080098e-03,  2.91034424e-02,  4.77284407e-01]
    #size_data = 7; input_data = [ 0.36517213, -0.04857957,  1.9047198 ,  0.378896  , -0.12225036, 0.02060622,  0.04555638]; output_data = [ 3.21080098e-03,  2.91034424e-02,  4.77284407e-01]
    
    visualize_shape_given_data(shapename, input_data = input_data, size_data = size_data, do_plot = True, color = 'b')
    visualize_shape_given_data(shapename, input_data = input_data, size_data = size_data, output_data = output_data, do_plot = True, color = 'c')
