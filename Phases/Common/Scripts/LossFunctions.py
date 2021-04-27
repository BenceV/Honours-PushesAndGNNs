import tensorflow as tf
from graph_nets import utils_tf
import numpy as np

def velocity_loss_single_step(target_op, output_op):
    """Create supervised loss operations from targets and outputs.

    Args:
        target_op: The target velocity tf.Tensor.
        output_ops: The list of output graphs from the model.

    Returns:
        A list of loss values (tf.Tensor), one per output op."""
    loss_op = [
            tf.reduce_mean(
            tf.reduce_sum(((output_op.nodes - target_op.nodes[..., 2:4])**2), axis=-1))
    ]
    return loss_op

def velocity_loss_single_roll(target_op, output_op):
    """Create supervised loss operations from targets and outputs.

    Args:
        target_op: The target velocity tf.Tensor.
        output_ops: The list of output graphs from the model.

    Returns:
        A list of loss values (tf.Tensor), one per output op."""
    loss_op = [
            tf.reduce_mean(
            tf.reduce_sum(((output_op.nodes[..., 2:4] - target_op.nodes[..., 2:4])**2), axis=-1))
    ]
    return loss_op

def velocity_error_single_roll(target_op, output_op):
    """Create supervised loss operations from targets and outputs.

    Args:
        target_op: The target velocity tf.Tensor.
        output_ops: The list of output graphs from the model.

    Returns:
        A list of loss values (tf.Tensor), one per output op."""
    loss_op = [
            np.sum(((output_op.nodes[..., 2:4] - target_op.nodes[..., 2:4])**2), axis=-1)
    ]
    return loss_op

def position_loss_single_step(target_op, output_op):
    """Create supervised loss operations from targets and outputs.

    Args:
        target_op: The target velocity tf.Tensor.
        output_ops: The list of output graphs from the model.

    Returns:
        A list of loss values (tf.Tensor), one per output op."""
    loss_op = [
            tf.reduce_mean(
            tf.reduce_sum(((output_op.nodes - target_op.nodes[..., 0:2])**2), axis=-1))
    ]
    return loss_op

def position_loss_single_roll(target_op, output_op):
    """Create supervised loss operations from targets and outputs.

    Args:
        target_op: The target position tf.Tensor.
        output_ops: The list of output graphs from the model.

    Returns:
        A list of loss values (tf.Tensor), one per output op."""
    loss_op = [
            tf.reduce_mean(
            tf.reduce_sum(((output_op.nodes[..., 0:2] - target_op.nodes[..., 0:2])**2), axis=-1))
    ]
    return loss_op

def position_error_single_roll(target_op, output_op):
    """Create supervised loss operations from targets and outputs.

    Args:
        target_op: The target velocity tf.Tensor.
        output_ops: The list of output graphs from the model.

    Returns:
        A list of loss values (tf.Tensor), one per output op."""
    loss_op = [
            np.sum(((output_op.nodes[...,0:2] - target_op.nodes[..., 0:2])**2), axis=-1)
    ]
    return loss_op


def make_all_runnable_in_session(*args):
    """Apply make_runnable_in_session to an iterable of graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]




def rollout_loss_mean_velocity(X, Y):
    losses = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        losses.append(velocity_loss_single_roll(x,y))
        
    losses = tf.convert_to_tensor(losses, dtype=tf.float32)
    return tf.math.reduce_mean(losses,axis=0)

def rollout_loss_max_velocity(X, Y):
    losses = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        losses.append(velocity_loss_single_roll(x,y))
        
    losses = tf.convert_to_tensor(losses, dtype=tf.float32)
    return tf.math.reduce_max(losses, axis=0)
    
def rollout_loss_sum_velocity(X, Y):
    losses = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        losses.append(velocity_loss_single_roll(x,y))
        
    losses = tf.convert_to_tensor(losses, dtype=tf.float32)
    return tf.math.reduce_sum(losses, axis=0)

def rollout_loss_mean_position(X, Y):
    losses = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        losses.append(position_loss_single_roll(x,y))
        
    losses = tf.convert_to_tensor(losses, dtype=tf.float32)
    return tf.math.reduce_mean(losses,axis=0)

def rollout_loss_max_position(X, Y):
    losses = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        losses.append(position_loss_single_roll(x,y))
        
    losses = tf.convert_to_tensor(losses, dtype=tf.float32)
    return tf.math.reduce_max(losses, axis=0)
    
def rollout_loss_sum_position(X, Y):
    losses = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        losses.append(position_loss_single_roll(x,y))
        
    losses = tf.convert_to_tensor(losses, dtype=tf.float32)
    return tf.math.reduce_sum(losses, axis=0)


def rollout_error_velocity(X, Y):
    losses = []
    steps = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        steps.append(i)
        losses.append(velocity_error_single_roll(y,x))
        
    losses = np.array(losses)
    steps = np.array(steps)
    return steps, losses


def rollout_error_position(X, Y):
    losses = []
    steps = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        steps.append(i)
        losses.append(position_error_single_roll(y,x))
        
    losses = np.array(losses)
    steps = np.array(steps)
    return steps, losses
