import numpy as np


def base_graph(n, both_ends, spring_constant, d, dampening):
    """Define a basic mass-spring system graph structure.

    These are n masses (1kg) connected by springs in a chain-like structure. The
    first and last masses are fixed. The masses are vertically aligned at the
    start and are d meters apart; this is also the rest length for the springs
    connecting them. Springs have spring constant spring_constant and gravity is 10 N in
    the negative y-direction.

    Args:
        #n - number of masses
        #spring_constant - the spring constant in N/m
        #d - distance between masses (as well as springs' rest length)
        #dampening - the dampening ratio

    Returns:
        data_dict - dictionary with globals, nodes, edges, receivers and senders
        to represent a structure like the one above."""

    # Nodes
    # Generate initial position and velocity for all masses.
    # The left-most mass has is at position (0, 0); other masses (ordered left to
    # right) have x-coordinate d meters apart from their left neighbor, and
    # y-coordinate 0. All masses have initial velocity 0m/s.

    # each node has an attribute vector with 6 dimensions, representing:
    #  [x_coord, y_coord, dx_coord, dy_coord, binary(fixed or not),kinetic_energy]
    #     nodes = np.zeros((n, 5), dtype=np.float32)
    nodes = np.zeros((n, 6), dtype=np.float64)

    # set the initial position of the masses(nodes) along the x axis
    half_width = d * n / 2.0
    nodes[:, 0] = np.linspace(-half_width, half_width, num=n, endpoint=False, dtype=np.float64)

    # indicate that the first and last masses are fixed
    if (both_ends):
        nodes[(0, -1), -2] = 1.
    else:
        nodes[0, -2] = 1.

    # Edges.
    edges, senders, receivers = [], [], []
    for i in range(n - 1):
        left_node = i
        right_node = i + 1
        # The 'if' statements prevent incoming edges to fixed ends of the string.
        # This is the reason why the system breaks when only one mass is fixed
        if right_node <= n - 1:
            # Left incoming edge - progagation from left to right
            edges.append([spring_constant, d, dampening, 0])  # atributes of the edge
            senders.append(left_node)  # index of the sender
            receivers.append(right_node)  # index of the receiver
        if left_node >= 0:
            # Right incoming edge - progagation from right to left
            edges.append([spring_constant, d, dampening, 0])  # atributes of the edge
            senders.append(right_node)  # index of the sender
            receivers.append(left_node)  # index of the receiver

    #  return a dictionary of the graph description
    return {
        "globals": np.float64([0., -10., 0.]),
        # 2-dim attributes of the graph's global, 0 force along x and -10 is gravity along y
        "nodes": np.float64(nodes),
        "edges": np.float64(edges),
        "receivers": receivers,
        "senders": senders}


def measure_errors_deviation_from_rectangle(positions):
    errors = []

    for position in positions:
        atlo1 = position[1, ...] - position[2, ...]
        atlo1 = np.sqrt(500) - np.sqrt(atlo1.dot(atlo1))
        atlo2 = position[0, ...] - position[3, ...]
        atlo2 = np.sqrt(500) - np.sqrt(atlo2.dot(atlo2))
        short1 = position[0, ...] - position[1, ...]
        short1 = 10 - np.sqrt(short1.dot(short1))
        short2 = position[2, ...] - position[3, ...]
        short2 = 10 - np.sqrt(short2.dot(short2))
        long1 = position[0, ...] - position[2, ...]
        long1 = 20 - np.sqrt(long1.dot(long1))
        long2 = position[1, ...] - position[3, ...]
        long2 = 20 - np.sqrt(long2.dot(long2))
        error_pre_sum = 1. / 2. * np.square(np.array([atlo1, atlo2, short1, short2, long1, long2]))
        errors = np.concatenate((errors, [np.sum(error_pre_sum)]))

    return errors

# Create a graph
def rectangle_graph(spring_constant, rest_lengths, dampening):
    """Define Basic dictionary of an n-body fully connected spring system

    The mass of these objects is  1 kg.
    The rest length of all springs is rest_lengths and all springs have a spring constant of spring_constant.
    All initials velocities are 0m/s.

    Arg:
        #n - the number of objects
        #spring_constant - the spring constant in N/m
        #rest_lengths - the rest lengths in m
        #indeces_of_fixed - a list consisting of the indeces of the nodes that are fixed
        #dampening - the dampening ratio

    #Returns:
        #dict - dictionary of the graph"""
    n = 5

    # Nodes
    # Each node has an x,y position and an x_v,y_v velocity
    nodes = np.zeros((n, 6), dtype=np.float64)
    # sampl = np.random.uniform(low=-10, high=10,size=(n,2))
    nodes[0, 0:2] = [-10, -5]
    nodes[1, 0:2] = [-10, 5]
    nodes[2, 0:2] = [10, -5]
    nodes[3, 0:2] = [10, 5]

    # nodes[:,0:2] = sampl
    nodes[4, 4] = 1

    # Edges
    # This is fully connected graph
    edges = np.array([spring_constant, rest_lengths, dampening, 0] * (n - 1) * (n - 2), dtype=np.float64).reshape(
        ((n - 1) * (n - 2), 4))
    senders = []
    receivers = []
    for i in range(n - 1):
        senders.extend([i] * (n - 2))  # index of the sender
        var = list(range(n))
        var.remove(i)
        var.remove(n - 1)
        receivers.extend(var)  # index of the receiver

    senders.extend([0, n - 1])  # node 0 sends an edge, node n-1 sends an edge
    receivers.extend([n - 1, 0])  # node n-1, the fixed, receives it,  node 0 receives it
    edges = np.concatenate((edges, np.array([50., rest_lengths, dampening, 0] * 2, dtype=np.float64).reshape(2, 4)),
                           axis=0)

    return {
        "globals": np.float64([0., -10., 0]),  # 3rd dimension of global feature is potential energy
        "nodes": np.float64(nodes),
        "edges": np.float64(edges),
        "receivers": receivers,
        "senders": senders}
# Create a graph
def create_graph(n, spring_constant, rest_lengths, indeces_of_fixed, dampening):
    """Define Basic dictionary of an n-body fully connected spring system

    The mass of these objects is  1 kg.
    The distance between these objects is random but is inside a square with lenght of its sides = 20 and its center in 0.
    The rest length of all springs is rest_lengths and all springs have a spring constant of spring_constant.
    All initials velocities are 0m/s.

    Arg:
        #n - the number of objects
        #spring_constant - the spring constant in N/m
        #rest_lengths - the rest lengths in m
        #indeces_of_fixed - a list consisting of the indeces of the nodes that are fixed
        #dampening - the dampening ratio

    #Returns:
        #dict - dictionary of the graph"""

    # Nodes
    # Each node has an x,y position and an x_v,y_v velocity
    nodes = np.zeros((n, 6), dtype=np.float64)
    sampl = np.random.uniform(low=-10, high=10, size=(n, 2))
    nodes[:, 0:2] = sampl
    if (not (indeces_of_fixed == [])):
        for i in indeces_of_fixed:
            nodes[i, 4] = 1

    # Edges
    # This is fully connected graph
    edges = np.array([spring_constant, rest_lengths, dampening, 0] * n * (n - 1), dtype=np.float64).reshape(
        (n * (n - 1), 4))
    senders = []
    receivers = []
    for i in range(n):
        senders.extend([i] * (n - 1))  # index of the sender
        var = list(range(n))
        var.remove(i)
        receivers.extend(var)  # index of the receiver

    return {
        "globals": np.float64([0., -10., 0]),  # 3rd dimension of global feature is potential energy
        "nodes": np.float64(nodes),
        "edges": np.float64(edges),
        "receivers": receivers,
        "senders": senders}