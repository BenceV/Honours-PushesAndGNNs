#Import usual things required for graph nets
import tensorflow as tf
from graph_nets import utils_tf

def create_loss_ops_velocity(target_op, output_ops):
    """Create supervised loss operations from targets and outputs.

    Args:
        target_op: The target velocity tf.Tensor.
        output_ops: The predicted nodes from the model.

    Returns:
        A list of loss values (tf.Tensor), one per output op."""
   
    loss_ops = [
            tf.reduce_mean(
            tf.reduce_sum((tf.cast(output_op,tf.float64) - tf.cast(target_op[..., 2:4],tf.float64))**2, axis=-1))
            for output_op in output_ops
    ]
    return loss_ops

def create_loss_op_velocity(target_op, output_op):
    """Create supervised loss operations from targets and outputs.

    Args:
        target_op: The target velocity tf.Tensor.
        output_op: The predicted nodes from the model.

    Returns:
        A list of loss values (tf.Tensor), one per output op."""
   
    loss_op = [
            tf.reduce_mean(
            tf.reduce_sum((tf.cast(output_op,tf.float64) - tf.cast(target_op[..., 2:4],tf.float64))**2, axis=-1))
    ]
    return loss_op




def create_loss_ops_position(target_op, output_ops):
    """Create supervised loss operations from targets and outputs.

    Args:
        target_op: The target velocity tf.Tensor.
        output_ops: The list of output graphs from the model.

    Returns:
        A list of loss values (tf.Tensor), one per output op."""
   
    loss_ops = [
            tf.reduce_mean(
            tf.reduce_sum((tf.cast(output_op.nodes,tf.float64) - tf.cast(target_op[..., 0:2],tf.float64))**2, axis=-1))
            for output_op in output_ops
    ]
    return loss_ops

def create_loss_op_position(target_op, output_op):
    """Create supervised loss operations from targets and outputs.

    Args:
        target_op: The target velocity tf.Tensor.
        output_ops: The list of output graphs from the model.

    Returns:
        A list of loss values (tf.Tensor), one per output op."""
   
    loss_op = [
            tf.reduce_mean(
            tf.reduce_sum((tf.cast(output_op,tf.float64) - tf.cast(target_op[..., 0:2],tf.float64))**2, axis=-1))
    ]
    return loss_op





def make_all_runnable_in_session(*args):
    """Apply make_runnable_in_session to an iterable of graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]