
import sonnet as snt
import tensorflow as tf
from graph_nets import blocks
def hookes_law(receiver_nodes, sender_nodes, k, x_rest):
    """Applies Hooke's law to springs connecting some nodes.

    Args:
        receiver_nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for the
        receiver node of each edge.
        sender_nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for the
        sender node of each edge.
        k: Spring constant for each edge.
        x_rest: Rest length of each edge.

    Returns:
        Nx2 Tensor of the force [f_x, f_y] acting on each edge."""
    diff = receiver_nodes[..., 0:2] - sender_nodes[..., 0:2]
    # diff is the vector difference between the two nodes
    x = tf.norm(diff, axis=-1, keepdims=True)
    # x is the distance between the two nodes
    force_magnitude = -1. * tf.multiply(k, (x - x_rest) / x)
    # The reason why we have the x there is to normalize for the next line, in which we build the force vector based on the disposition
    force = force_magnitude * diff
    return force


def damping_force(receiver_nodes, sender_nodes, d):
    """Calculates the damping force between connected nodes.

    Args:
        receiver_nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for the
        receiver node of each edge.
        sender_nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for the
        sender node of each edge.
        d: Damping ratio of each edge.

    Returns:
        Nx2 Tensor of the force [f_x, f_y] acting on each node."""
    velocity = receiver_nodes[..., 2:4]
    force = tf.multiply(d, velocity)
    return force


def calculate_energy_of_system(graph, velocities, receiver_nodes, sender_nodes, k, x_rest):
    """Calculate the potential energy of the whole system.

    Args:
        graph
        velocities: Ex2 tf.Tensor of [v_x, v_y] features for every node.
        receiver_nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for the
        receiver node of each edge.
        sender_nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for the
        sender node of each edge.
        k: Spring constant for each edge.
        x_rest: Rest length of each edge.

    Returns:
        potential energy: potential energy of the system """
    # This reducer when supplied to an aggregator will sum up the incoming values from the corresponding graphs
    reducer2 = tf.unsorted_segment_sum

    # print("Shape of velocities: "+str(velocities.shape))
    # print("Shape of receiver_nodes: "+str(receiver_nodes.shape))
    # print("Shape of sender_nodes: "+str(sender_nodes.shape))
    # print("Shape of k: "+str(k.shape))
    # print("Shape of x_rest: "+str(x_rest.shape))

    # print("Shape of globals graph: "+str(graph.globals.shape))
    # print("Shape of nodes graph: "+str(graph.nodes.shape))
    # print("Shape of edges graph: "+str(graph.edges.shape))

    # Create a copy of the graph
    energy_graph = graph.replace(globals=tf.zeros((1, 1)))

    kinetic_energy = (1. / 2.) * tf.square(tf.norm(velocities, axis=-1, keepdims=True))

    diff = receiver_nodes[..., 0:2] - sender_nodes[..., 0:2]
    x = tf.norm(diff, axis=-1, keepdims=True)
    spring_energy = (1. / 2.) * tf.multiply(k, tf.abs((x - x_rest)))

    # potential_energy = receiver_nodes[...,1]*9.8
    # Replace the graph edges, with spring energy
    # energy_graph = energy_graph.replace(edges=potential_energy)
    # Aggregate spring energy to global
    # pot_en = blocks.EdgesToGlobalsAggregator(reducer=reducer2)(energy_graph)
    # Replace the graph edges, with spring energy
    energy_graph = energy_graph.replace(edges=spring_energy)
    # Aggregate spring energy to global
    spr_en = blocks.EdgesToGlobalsAggregator(reducer=reducer2)(energy_graph)
    # Replace the graph edges, with kinetic energy
    energy_graph = energy_graph.replace(edges=kinetic_energy)
    # Aggregate kinetic energy to global
    kin_en = blocks.EdgesToGlobalsAggregator(reducer=reducer2)(energy_graph) / 2.

    # print("Shape of globals energy_graph: "+str(energy_graph.globals.shape))
    # print("Shape of nodes energy_graph: "+str(energy_graph.nodes.shape))
    # print("Shape of edges energy_graph: "+str(energy_graph.edges.shape))

    # print("spring energy shape"+str(spr_en.shape))
    # print("kinetic energy shape"+str(kin_en.shape))
    # print("potential energy shape"+str(pot_en.shape))

    energy = spr_en + kin_en  # +pot_en
    # print("shape of energy"+str(energy.shape))

    return energy


def euler_integration(nodes, force_per_node, step_size):
    """Applies one step of Euler integration.

    Args:
        nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for each node.
        force_per_node: Ex2 tf.Tensor of the force [f_x, f_y] acting on each edge.
        step_size: Scalar.

    Returns:
        A tf.Tensor of the same shape as `nodes` but with positions and velocities
        updated."""
    is_fixed_to_move = nodes[..., 4:5]
    # set forces to zero for fixed nodes
    force_per_node *= 1. - is_fixed_to_move
    new_vel = nodes[..., 2:4] + force_per_node * step_size
    return new_vel


class SpringMassSimulator(snt.AbstractModule):
    """Implements a basic Physics Simulator using the blocks library."""

    def __init__(self, step_size, name="SpringMassSimulator"):
        super(SpringMassSimulator, self).__init__(name=name)
        self._step_size = step_size

        with self._enter_variable_scope():
            self._aggregator = blocks.ReceivedEdgesToNodesAggregator(
                reducer=tf.unsorted_segment_sum)

    def _build(self, graph):
        """Builds a SpringMassSimulator.
        Args:
        graph: A graphs.GraphsTuple having, for some integers N, E, G:
              - edges: Nx2 tf.Tensor of [spring_constant, rest_length] for each
            edge.
              - nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for each
            node.
              - globals: Gx3 tf.Tensor containing the gravitational constant and the potential energy of the system

        Returns:
          A graphs.GraphsTuple of the same shape as `graph`, but where:
              - edges: Holds the force [f_x, f_y] acting on each edge.
              - nodes: Holds positions and velocities after applying one step of
              Euler integration.
        """
        receiver_nodes = blocks.broadcast_receiver_nodes_to_edges(graph)
        sender_nodes = blocks.broadcast_sender_nodes_to_edges(graph)

        spring_force_per_edge = hookes_law(receiver_nodes, sender_nodes,
                                           graph.edges[..., 0:1],
                                           graph.edges[..., 1:2])

        spring_force_per_edge -= damping_force(receiver_nodes, sender_nodes, graph.edges[..., 2:3])

        velocities = blocks.broadcast_receiver_nodes_to_edges(graph)[..., 2:4]

        energy = calculate_energy_of_system(
            graph,
            velocities,
            receiver_nodes,
            sender_nodes,
            graph.edges[..., 0:1],
            graph.edges[..., 1:2])

        graph = graph.replace(edges=spring_force_per_edge)
        spring_force_per_node = self._aggregator(graph)

        gravity = blocks.broadcast_globals_to_nodes(graph)[..., 0:2]
        updated_velocities = euler_integration(graph.nodes, spring_force_per_node + gravity, self._step_size)

        globs = tf.concat([graph.globals[..., 0:2], energy], axis=-1)
        graph = graph.replace(globals=globs)

        graph = graph.replace(nodes=updated_velocities)
        # print("Shape of globals: "+str(graph.globals.shape))
        # print("Shape of nodes: "+str(graph.nodes.shape))
        # print("Shape of edges: "+str(graph.edges.shape))

        return graph


def prediction_to_next_state(input_graph, predicted_graph, step_size):
    # manually integrate velocities to compute new positions
    new_pos = input_graph.nodes[..., :2] + predicted_graph.nodes * step_size
    new_nodes = tf.concat([new_pos, predicted_graph.nodes, input_graph.nodes[..., 4:5]], axis=-1)

    input_graph = input_graph.replace(nodes=new_nodes)

    globs = tf.concat([input_graph.globals[...,0:2],predicted_graph.globals[...,2:3]],axis=-1)
    input_graph = input_graph.replace(globals=globs)

    return input_graph


def roll_out_physics(simulator, graph, steps, step_size):
    """Apply some number of steps of physical laws to an interaction network.

    Args:
        simulator: A SpringMassSimulator, or some module or callable with the same
        signature.
        graph: A graphs.GraphsTuple having, for some integers N, E, G:
            - edges: Nx3 tf.Tensor of [spring_constant, rest_length, damping_ratio] for each edge.
            - nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for each
              node.
            - globals: Gx3 tf.Tensor containing the gravitational constant and the potential energy of the system
        steps: An integer.
        step_size: Scalar.

    Returns:
        A pair of:
        - The graph, updated after `steps` steps of simulation;
        - A `steps+1`xNx5 tf.Tensor of the node features at each step."""

    def body(t, graph, nodes_per_step, globals_per_step):
        predicted_graph = simulator(graph)
        if isinstance(predicted_graph, list):
            predicted_graph = predicted_graph[-1]
        graph = prediction_to_next_state(graph, predicted_graph, step_size)
        return t + 1, graph, nodes_per_step.write(t, graph.nodes), globals_per_step.write(t, graph.globals)

    globals_per_step = tf.TensorArray(
        dtype=graph.globals.dtype, size=steps + 1, element_shape=graph.globals.shape)
    globals_per_step = globals_per_step.write(0, graph.globals)

    nodes_per_step = tf.TensorArray(
        dtype=graph.nodes.dtype, size=steps + 1, element_shape=graph.nodes.shape)
    nodes_per_step = nodes_per_step.write(0, graph.nodes)

    _, g, nodes_per_step, globals_per_step = tf.while_loop(
        lambda t, *unused_args: t <= steps,
        body,
        loop_vars=[1, graph, nodes_per_step, globals_per_step])
    return g, nodes_per_step.stack(), globals_per_step.stack()


def apply_noise(graph, node_noise_level, edge_noise_level, global_noise_level):
    """Applies uniformly-distributed noise to a graph of a physical system.

    Noise is applied to:
    - the x and y coordinates (independently) of the nodes;
    - the spring constants of the edges;
    - the y coordinate of the global gravitational constant.

    Args:
        graph: a graphs.GraphsTuple having, for some integers N, E, G:
            - nodes: Nx5 Tensor of [x, y, _, _, _] for each node.
            - edges: Ex2 Tensor of [spring_constant, _] for each edge.
            - globals: Gx3 tf.Tensor containing the gravitational constant and the potential energy of the system
        node_noise_level: Maximum distance to perturb nodes' x and y coordinates.
        edge_noise_level: Maximum amount to perturb edge spring constants.
        global_noise_level: Maximum amount to perturb the Y component of gravity.

    Returns:
        The input graph, but with noise applied."""
    node_position_noise = tf.random_uniform(
        [graph.nodes.shape[0].value, 2],
        minval=-node_noise_level,
        maxval=node_noise_level, dtype=tf.float64)
    edge_spring_constant_noise = tf.random_uniform(
        [graph.edges.shape[0].value, 1],
        minval=-edge_noise_level,
        maxval=edge_noise_level, dtype=tf.float64)
    global_gravity_y_noise = tf.random_uniform(
        [graph.globals.shape[0].value, 1],
        minval=-global_noise_level,
        maxval=global_noise_level, dtype=tf.float64)

    return graph.replace(
        nodes=tf.concat(
            [graph.nodes[..., :2] + node_position_noise,
             graph.nodes[..., 2:]],
            axis=-1),
        edges=tf.concat(
            [
                graph.edges[..., 0:1] + edge_spring_constant_noise,
                graph.edges[..., 1:2],
                graph.edges[..., 2:3]
            ],
            axis=-1),
        globals=tf.concat(
            [
                graph.globals[..., 0:1],
                graph.globals[..., 1:2] + global_gravity_y_noise,
                graph.globals[..., 2:3]
            ],
            axis=-1))


def set_rest_lengths(graph):
    """Computes and sets rest lengths for the springs in a physical system.

    The rest length is taken to be the distance between each edge's nodes.

    Args:
        graph: a graphs.GraphsTuple having, for some integers N, E:
            - nodes: Nx5 Tensor of [x, y, _, _, _] for each node.
            - edges: Ex2 Tensor of [spring_constant, _] for each edge.

    Returns:
        The input graph, but with [spring_constant, rest_length] for each edge."""
    receiver_nodes = blocks.broadcast_receiver_nodes_to_edges(graph)
    sender_nodes = blocks.broadcast_sender_nodes_to_edges(graph)
    rest_length = tf.norm(
        receiver_nodes[..., :2] - sender_nodes[..., :2], axis=-1, keepdims=True)
    return graph.replace(
        edges=tf.concat([graph.edges[..., :1], rest_length, graph.edges[..., 2:3]], axis=-1))


def generate_trajectory(simulator, graph, steps, step_size, node_noise_level,
                        edge_noise_level, global_noise_level, do_set_rest, do_apply_gravity):
    """Applies noise and then simulates a physical system for a number of steps.

    Args:
        simulator: A SpringMassSimulator, or some module or callable with the same
          signature.
        graph: a graphs.GraphsTuple having, for some integers N, E, G:
            - nodes: Nx5 Tensor of [x, y, v_x, v_y, is_fixed] for each node.
            - edges: Ex3 Tensor of [spring_constant, _] for each edge.
            - globals: Gx3 tf.Tensor containing the gravitational constant and the potential energy of the system
    steps: Integer; the length of trajectory to generate.
    step_size: Scalar.
    node_noise_level: Maximum distance to perturb nodes' x and y coordinates.
    edge_noise_level: Maximum amount to perturb edge spring constants.
    global_noise_level: Maximum amount to perturb the Y component of gravity.

  Returns:
    A pair of:
        - The input graph, but with rest lengths computed and noise applied.
        - A `steps+1`xNx5 tf.Tensor of the node features at each step."""
    graph = apply_noise(graph, node_noise_level, edge_noise_level,
                        global_noise_level)
    if (do_set_rest):
        graph = set_rest_lengths(graph)

    if (not (do_apply_gravity)):
        graph = graph.replace(globals=tf.zeros(graph.globals.shape, dtype=tf.float32))

    _, nodes_per_step, globals_per_step = roll_out_physics(simulator, graph, steps, step_size)
    return graph, nodes_per_step, globals_per_step

