import numpy as np


# Create a graph
def chain_graph_simple_one_way(n, d):
    """Define a basic mass-spring system graph structure.

        These are n masses (1kg) connected by springs in a chain-like structure. The
        first and last masses are fixed. The masses are vertically aligned at the
        start and are d meters apart; this is also the rest length for the springs
        connecting them. Springs have spring constant spring_constant and gravity is 10 N in
        the negative y-direction.

        Args:
            #n - number of masses
            #d - distance between masses (as well as springs' rest length)

        Returns:
            data_dict - dictionary with globals, nodes, edges, receivers and senders
            to represent a structure like the one above."""

    # Nodes
    # Generate initial position and velocity for all masses.
    # The left-most mass has is at position (0, 0); other masses (ordered left to
    # right) have x-coordinate d meters apart from their left neighbor, and
    # y-coordinate 0. All masses have initial velocity 0m/s.

    # each node has an attribute vector with 5 dimensions, representing:
    #  [x_coord, y_coord, dx_coord, dy_coord, binary(fixed or not)]
    nodes = np.zeros((n, 5), dtype=np.float32)

    # set the initial position of the masses(nodes) along the x axis
    half_width = d * n / 2.0
    nodes[:, 0] = np.linspace(-half_width, half_width, num=n, endpoint=False, dtype=np.float64)

    # indicate that the first and last masses are fixed
    nodes[(0, -1), -1] = 1.

    # Edges.
    edges, senders, receivers = [], [], []
    for i in range(n - 1):
        left_node = i
        right_node = i + 1
        # The 'if' statements prevent incoming edges to fixed ends of the string.
        # This is the reason why the system breaks when only one mass is fixed
        if right_node < n - 1:
            # Left incoming edge - progagation from left to right
            edges.append([50, d])  # atributes of the edge
            senders.append(left_node)  # index of the sender
            receivers.append(right_node)  # index of the receiver
        if left_node > 0:
            # Right incoming edge - progagation from right to left
            edges.append([50, d])  # atributes of the edge
            senders.append(right_node)  # index of the sender
            receivers.append(left_node)  # index of the receiver

    #  return a dictionary of the graph description
    return {
        "globals": np.float32([0., -10.]),
        # 2-dim attributes of the graph's global, 0 force along x and -10 is gravity along y
        "nodes": np.float32(nodes),
        "edges": np.float32(edges),
        "receivers": receivers,
        "senders": senders}


def chain_graph_simple_two_way(n, d):
    """Define a basic mass-spring system graph structure.

        These are n masses (1kg) connected by springs in a chain-like structure. The
        first and last masses are fixed. The masses are vertically aligned at the
        start and are d meters apart; this is also the rest length for the springs
        connecting them. Springs have spring constant spring_constant and gravity is 10 N in
        the negative y-direction.

        Args:
            #n - number of masses
            #d - distance between masses (as well as springs' rest length)

        Returns:
            data_dict - dictionary with globals, nodes, edges, receivers and senders
            to represent a structure like the one above."""

    # Nodes
    # Generate initial position and velocity for all masses.
    # The left-most mass has is at position (0, 0); other masses (ordered left to
    # right) have x-coordinate d meters apart from their left neighbor, and
    # y-coordinate 0. All masses have initial velocity 0m/s.

    # each node has an attribute vector with 5 dimensions, representing:
    #  [x_coord, y_coord, dx_coord, dy_coord, binary(fixed or not)]
    nodes = np.zeros((n, 5), dtype=np.float32)

    # set the initial position of the masses(nodes) along the x axis
    half_width = d * n / 2.0
    nodes[:, 0] = np.linspace(-half_width, half_width, num=n, endpoint=False, dtype=np.float64)

    # indicate that the first and last masses are fixed
    nodes[(0, -1), -1] = 1.

    # Edges.
    edges, senders, receivers = [], [], []
    for i in range(n - 1):
        left_node = i
        right_node = i + 1
        # The 'if' statements prevent incoming edges to fixed ends of the string.
        # This is the reason why the system breaks when only one mass is fixed
        if right_node <= n - 1:
            # Left incoming edge - progagation from left to right
            edges.append([50, d])  # atributes of the edge
            senders.append(left_node)  # index of the sender
            receivers.append(right_node)  # index of the receiver
        if left_node >= 0:
            # Right incoming edge - progagation from right to left
            edges.append([50, d])  # atributes of the edge
            senders.append(right_node)  # index of the sender
            receivers.append(left_node)  # index of the receiver

    #  return a dictionary of the graph description
    return {
        "globals": np.float32([0., -10.]),
        # 2-dim attributes of the graph's global, 0 force along x and -10 is gravity along y
        "nodes": np.float32(nodes),
        "edges": np.float32(edges),
        "receivers": receivers,
        "senders": senders}




def chain_graph_one_ended_simple(n, d):
    """Define a basic mass-spring system graph structure.

        These are n masses (1kg) connected by springs in a chain-like structure. The
        first and last masses are fixed. The masses are vertically aligned at the
        start and are d meters apart; this is also the rest length for the springs
        connecting them. Springs have spring constant spring_constant and gravity is 10 N in
        the negative y-direction.

        Args:
            #n - number of masses
            #d - distance between masses (as well as springs' rest length)

        Returns:
            data_dict - dictionary with globals, nodes, edges, receivers and senders
            to represent a structure like the one above."""

    # Nodes
    # Generate initial position and velocity for all masses.
    # The left-most mass has is at position (0, 0); other masses (ordered left to
    # right) have x-coordinate d meters apart from their left neighbor, and
    # y-coordinate 0. All masses have initial velocity 0m/s.

    # each node has an attribute vector with 5 dimensions, representing:
    #  [x_coord, y_coord, dx_coord, dy_coord, binary(fixed or not)]
    nodes = np.zeros((n, 5), dtype=np.float32)

    # set the initial position of the masses(nodes) along the x axis
    half_width = d * n / 2.0
    nodes[:, 0] = np.linspace(-half_width, half_width, num=n, endpoint=False, dtype=np.float32)

    # indicate that the first mass is fixed
    nodes[0, -1] = 1.

    # Edges.
    edges, senders, receivers = [], [], []
    for i in range(n - 1):
        left_node = i
        right_node = i + 1
        if right_node <= n - 1:
            # Left incoming edge - progagation from left to right
            edges.append([50, d])  # atributes of the edge
            senders.append(left_node)  # index of the sender
            receivers.append(right_node)  # index of the receiver
        if left_node > 0:
            # Right incoming edge - progagation from right to left
            edges.append([50, d])  # atributes of the edge
            senders.append(right_node)  # index of the sender
            receivers.append(left_node)  # index of the receiver

    #  return a dictionary of the graph description
    return {
        "globals": np.float32([0., -10.]),
        # 2-dim attributes of the graph's global, 0 force along x and -10 is gravity along y
        "nodes": np.float32(nodes),
        "edges": np.float32(edges),
        "receivers": receivers,
        "senders": senders}


# Create a graph
def chain_graph_one_ended_extended_one_way(n, spring_constant, d, dampening):
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

    # each node has an attribute vector with 5 dimensions, representing:
    #  [x_coord, y_coord, dx_coord, dy_coord, binary(fixed or not)]
    #     nodes = np.zeros((n, 5), dtype=np.float32)
    nodes = np.zeros((n, 5), dtype=np.float32)

    # set the initial position of the masses(nodes) along the x axis
    half_width = d * n / 2.0
    nodes[:, 0] = np.linspace(-half_width, half_width, num=n, endpoint=False, dtype=np.float32)

    # indicate that the first mass is fixed

    nodes[0, -1] = 1.

    # Edges.
    edges, senders, receivers = [], [], []
    for i in range(n - 1):
        left_node = i
        right_node = i + 1
        # The 'if' statements prevent incoming edges to fixed ends of the string.
        # This is the reason why the system breaks when only one mass is fixed
        if right_node <= n - 1:
            # Left incoming edge - progagation from left to right
            edges.append([spring_constant, d, dampening])  # atributes of the edge
            senders.append(left_node)  # index of the sender
            receivers.append(right_node)  # index of the receiver
        if left_node > 0:
            # Right incoming edge - progagation from right to left
            edges.append([spring_constant, d, dampening])  # atributes of the edge
            senders.append(right_node)  # index of the sender
            receivers.append(left_node)  # index of the receiver

    #  return a dictionary of the graph description
    return {
        "globals": np.float32([0., -10.]),
        # 2-dim attributes of the graph's global, 0 force along x and -10 is gravity along y
        "nodes": np.float32(nodes),
        "edges": np.float32(edges),
        "receivers": receivers,
        "senders": senders}

# Create a graph
def chain_graph_extended_one_way(n, spring_constant, d, dampening):
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

    # each node has an attribute vector with 5 dimensions, representing:
    #  [x_coord, y_coord, dx_coord, dy_coord, binary(fixed or not)]
    #     nodes = np.zeros((n, 5), dtype=np.float32)
    nodes = np.zeros((n, 5), dtype=np.float32)

    # set the initial position of the masses(nodes) along the x axis
    half_width = d * n / 2.0
    nodes[:, 0] = np.linspace(-half_width, half_width, num=n, endpoint=False, dtype=np.float32)

    # indicate that the first and last masses are fixed

    nodes[(0, -1), -1] = 1.

    # Edges.
    edges, senders, receivers = [], [], []
    for i in range(n - 1):
        left_node = i
        right_node = i + 1
        # The 'if' statements prevent incoming edges to fixed ends of the string.
        # This is the reason why the system breaks when only one mass is fixed
        if right_node < n - 1:
            # Left incoming edge - progagation from left to right
            edges.append([spring_constant, d, dampening])  # atributes of the edge
            senders.append(left_node)  # index of the sender
            receivers.append(right_node)  # index of the receiver
        if left_node > 0:
            # Right incoming edge - progagation from right to left
            edges.append([spring_constant, d, dampening])  # atributes of the edge
            senders.append(right_node)  # index of the sender
            receivers.append(left_node)  # index of the receiver

    #  return a dictionary of the graph description
    return {
        "globals": np.float32([0., -10.]),
        # 2-dim attributes of the graph's global, 0 force along x and -10 is gravity along y
        "nodes": np.float32(nodes),
        "edges": np.float32(edges),
        "receivers": receivers,
        "senders": senders}


def chain_graph_extended_two_way(n, spring_constant, d, dampening):
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

    # each node has an attribute vector with 5 dimensions, representing:
    #  [x_coord, y_coord, dx_coord, dy_coord, binary(fixed or not)]
    #     nodes = np.zeros((n, 5), dtype=np.float32)
    nodes = np.zeros((n, 5), dtype=np.float32)

    # set the initial position of the masses(nodes) along the x axis
    half_width = d * n / 2.0
    nodes[:, 0] = np.linspace(-half_width, half_width, num=n, endpoint=False, dtype=np.float32)

    # indicate that the first and last masses are fixed

    nodes[(0, -1), -1] = 1.

    # Edges.
    edges, senders, receivers = [], [], []
    for i in range(n - 1):
        left_node = i
        right_node = i + 1
        # The 'if' statements prevent incoming edges to fixed ends of the string.
        # This is the reason why the system breaks when only one mass is fixed
        if right_node <= n - 1:
            # Left incoming edge - progagation from left to right
            edges.append([spring_constant, d, dampening])  # atributes of the edge
            senders.append(left_node)  # index of the sender
            receivers.append(right_node)  # index of the receiver
        if left_node >= 0:
            # Right incoming edge - progagation from right to left
            edges.append([spring_constant, d, dampening])  # atributes of the edge
            senders.append(right_node)  # index of the sender
            receivers.append(left_node)  # index of the receiver

    #  return a dictionary of the graph description
    return {
        "globals": np.float32([0., -10., 0.]),
        # 2-dim attributes of the graph's global, 0 force along x and -10 is gravity along y
        "nodes": np.float32(nodes),
        "edges": np.float32(edges),
        "receivers": receivers,
        "senders": senders}




# Create a graph
def rigid_rectangle_graph(spring_constant, rest_lengths,one_way, damping):
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
    nodes = np.zeros((n, 5), dtype=np.float32)
    # sampl = np.random.uniform(low=-10, high=10,size=(n,2))
    nodes[0, 0:2] = [-10, -5]
    nodes[1, 0:2] = [-10, 5]
    nodes[2, 0:2] = [10, -5]
    nodes[3, 0:2] = [10, 5]

    # nodes[:,0:2] = sampl
    nodes[4, 4] = 1

    # Edges
    # This is fully connected graph
    edges = np.array([spring_constant, rest_lengths, damping] * (n - 1) * (n - 2), dtype=np.float32).reshape(
        ((n - 1) * (n - 2), 3))
    senders = []
    receivers = []
    for i in range(n - 1):
        senders.extend([i] * (n - 2))  # index of the sender
        var = list(range(n))
        var.remove(i)
        var.remove(n - 1)
        receivers.extend(var)  # index of the receiver

    if one_way:
        senders.extend([n - 1])  # fixed sends an edge
        receivers.extend([0])  # not fixed receives it
        edges = np.concatenate((edges, np.array([10., rest_lengths, damping], dtype=np.float32).reshape(1, 3)), axis=0)
    else:
        senders.extend([0, n - 1])  # node 0 sends an edge, node n-1 sends an edge
        receivers.extend([n - 1, 0])  # node n-1, the fixed, receives it,  node 0 receives it
        edges = np.concatenate((edges, np.array([10., rest_lengths, damping] * 2, dtype=np.float32).reshape(2, 3)),
                           axis=0)

    return {
        "globals": np.float32([0., -10.]),
        "nodes": np.float32(nodes),
        "edges": np.float32(edges),
        "receivers": receivers,
        "senders": senders}


def rigid_rectangle_graph_extended(len_a, len_b, spring_constant, one_way, damping, pos_a, pos_b, vel_a, vel_b, n_contacts=1):
    """Define Basic dictionary of an n-body fully connected spring system

    The mass of these objects is  1 kg.
    The rest length of all springs is rest_lengths and all springs have a spring constant of spring_constant.
    All initials velocities are 0m/s.

    Arg:
        #len_a - half of the distance between 2 node for an edge on the x axis 
        #len_b - half of the distance between 2 node for an edge on the y axis
        #spring_constant - the spring constant in N/m
        #rest_lengths - the rest lengths in m
        #indeces_of_fixed - a list consisting of the indeces of the nodes that are fixed
        #dampening - the dampening ratio

    #Returns:
        #dict - dictionary of the graph"""
    rest_lengths = 1.0
    n = 5
    # Nodes
    # Each node has an x,y position and an x_v,y_v velocity
    nodes = np.zeros((n, 5), dtype=np.float32)
    nodes[0, 0:2] = [-len_a, -len_b]
    nodes[1, 0:2] = [-len_a, len_b]
    nodes[2, 0:2] = [len_a, len_b]
    nodes[3, 0:2] = [len_a, -len_b]

    nodes[4, 0:4] = [pos_a, pos_b, vel_a, vel_b]
    nodes[4, 4] = 1

    distances = np.sqrt(np.square(nodes[:4, 0] - nodes[4, 0]) + np.square(nodes[:4, 1] - nodes[4, 1]))
    tup_distances = list(set(zip([0, 1, 2, 3], distances)))

    def helper_second(elem):
        return elem[1]

    tup_distances.sort(key=helper_second)
    tup_distances = tup_distances[:n_contacts]


    # Edges
    # This is fully connected graph
    edges = np.array([spring_constant, rest_lengths, damping] * (n - 1) * (n - 2), dtype=np.float32).reshape(
        ((n - 1) * (n - 2), 3))
    senders = []
    receivers = []
    for i in range(n - 1):
        senders.extend([i] * (n - 2))  # index of the sender
        var = list(range(n))
        var.remove(i)
        var.remove(n - 1)
        receivers.extend(var)  # index of the receiver

    for node_numb, _ in tup_distances:
        if one_way:
            senders.extend([n - 1])  # fixed sends an edge
            receivers.extend([node_numb])  # not fixed receives it
            edges = np.concatenate((edges, np.array([70, rest_lengths, damping], dtype=np.float32).reshape(1, 3)), axis=0)
        else:
            senders.extend([node_numb, n - 1])  # node 0 sends an edge,     node n-1 sends an edge
            receivers.extend([n - 1, node_numb])  # node n-1, the fixed, receives it,  node 0 receives it
            edges = np.concatenate((edges, np.array([70, rest_lengths, damping] * 2, dtype=np.float32).reshape(2, 3)), axis=0)

    return {
        "globals": np.float32([0., -10.]),  # 3rd dimension of global feature is potential energy
        "nodes": np.float32(nodes),
        "edges": np.float32(edges),
        "receivers": receivers,
        "senders": senders}


# Create a graph
def rigid_rectangle_graph_simple(spring_constant, rest_lengths, one_way):
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
    nodes = np.zeros((n, 5), dtype=np.float32)
    # sampl = np.random.uniform(low=-10, high=10,size=(n,2))
    nodes[0, 0:2] = [-10, -5]
    nodes[1, 0:2] = [-10, 5]
    nodes[2, 0:2] = [10, -5]
    nodes[3, 0:2] = [10, 5]

    # nodes[:,0:2] = sampl
    nodes[4, 4] = 1

    # Edges
    # This is fully connected graph
    edges = np.array([spring_constant, rest_lengths] * (n - 1) * (n - 2), dtype=np.float32).reshape(
        ((n - 1) * (n - 2), 2))
    senders = []
    receivers = []
    for i in range(n - 1):
        senders.extend([i] * (n - 2))  # index of the sender
        var = list(range(n))
        var.remove(i)
        var.remove(n - 1)
        receivers.extend(var)  # index of the receiver

    if one_way:
        senders.extend([n - 1])  # fixed sends an edge
        receivers.extend([0])   # not fixed receives it
        edges = np.concatenate((edges, np.array([10., rest_lengths], dtype=np.float32).reshape(1, 2)), axis=0)
    else:
        senders.extend([0, n - 1])  # node 0 sends an edge, node n-1 sends an edge
        receivers.extend([n - 1, 0])  # node n-1, the fixed, receives it,  node 0 receives it
        edges = np.concatenate((edges, np.array([10., rest_lengths] * 2, dtype=np.float32).reshape(2, 2)), axis=0)

    return {
        "globals": np.float32([0., -10.]),  # 3rd dimension of global feature is potential energy
        "nodes": np.float32(nodes),
        "edges": np.float32(edges),
        "receivers": receivers,
        "senders": senders}




# Create a graph
def rigid_polygon_graph(n, spring_constant, rest_lengths, indeces_of_fixed, damping):
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
    nodes = np.zeros((n, 5), dtype=np.float64)
    sampl = np.random.uniform(low=-10, high=10, size=(n, 2))
    nodes[:, 0:2] = sampl
    if (not (indeces_of_fixed == [])):
        for i in indeces_of_fixed:
            nodes[i, 4] = 1

    # Edges
    # This is fully connected graph
    edges = np.array([spring_constant, rest_lengths, damping] * n * (n - 1), dtype=np.float64).reshape(
        (n * (n - 1), 3))
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


def rigid_graph_from_pos(node_pos, one_way, tip_pos, tip_vel, n_contacts=1):
    """Define Basic dictionary of an n-body fully connected spring system

    The mass of these objects is  1 kg.
    The rest length of all springs is rest_lengths and all springs have a spring constant of spring_constant.
    All initials velocities are 0m/s.

    Arg:
        #len_a - half of the distance between 2 node for an edge on the x axis 
        #len_b - half of the distance between 2 node for an edge on the y axis
        #spring_constant - the spring constant in N/m
        #rest_lengths - the rest lengths in m
        #indeces_of_fixed - a list consisting of the indeces of the nodes that are fixed
        #dampening - the dampening ratio

    #Returns:
        #dict - dictionary of the graph"""
    # Nodes
    # Each node has an x,y position and an x_v,y_v velocity
    nodes = np.zeros((6, 5), dtype=np.float32)
    nodes[0, 0:2] = node_pos[0]
    nodes[1, 0:2] = node_pos[1]
    nodes[2, 0:2] = node_pos[2]
    nodes[3, 0:2] = node_pos[3]
    nodes[4, 0:2] = node_pos[4]

    nodes[5, 0:2] = tip_pos
    nodes[5, 2:4] = tip_vel
    nodes[5, 4] = 1

    distances = np.sqrt(np.square(nodes[:4, 0] - nodes[4, 0]) + np.square(nodes[:4, 1] - nodes[4, 1]))
    tup_distances = list(set(zip([0, 1, 2, 3], distances)))

    def helper_second(elem):
        return elem[1]

    tup_distances.sort(key=helper_second)
    tup_distances = tup_distances[:n_contacts]


    # Edges
    is_rigid = 1.0
    # This is fully connected graph
    edges = np.array([is_rigid, ] * (n - 1) * (n - 2), dtype=np.float32).reshape(
        ((n - 1) * (n - 2), 3))
    senders = []
    receivers = []
    for i in range(n - 1):
        senders.extend([i] * (n - 2))  # index of the sender
        var = list(range(n))
        var.remove(i)
        var.remove(n - 1)
        receivers.extend(var)  # index of the receiver

    for node_numb, _ in tup_distances:
        if one_way:
            senders.extend([n - 1])  # fixed sends an edge
            receivers.extend([node_numb])  # not fixed receives it
            edges = np.concatenate((edges, np.array([70, rest_lengths, damping], dtype=np.float32).reshape(1, 3)), axis=0)
        else:
            senders.extend([node_numb, n - 1])  # node 0 sends an edge,     node n-1 sends an edge
            receivers.extend([n - 1, node_numb])  # node n-1, the fixed, receives it,  node 0 receives it
            edges = np.concatenate((edges, np.array([70, rest_lengths, damping] * 2, dtype=np.float32).reshape(2, 3)), axis=0)

    return {
        "globals": np.float32([0., -10.]),  # 3rd dimension of global feature is potential energy
        "nodes": np.float32(nodes),
        "edges": np.float32(edges),
        "receivers": receivers,
        "senders": senders}



def rigid_graph_from_pos_closest(obj_pos, obj_vel, tip_pos, tip_vel, tip_con, do_cont = False):
    """
    Define dictionary containing the needed for building a 
    graph representation of the physical systems from the MIT push dataset.
    
    This function builds connectivity based on the following definition:
     - A rigid rectangle object consists of 5 nodes: The four corners of the object, and the center point of the object
     - The Object nodes are connected:
        -------------------------------------
        t_r (top_right, 0) <- t_l (top_left, 1)
        t_r (top_right, 0) <- b_r (bottom_right, 2)
        t_r (top_right, 0) <- m_m (center, 4)
        -------------------------------------
        t_l (top_left, 1) <- t_r (top_right, 0)
        t_l (top_left, 1) <- b_l (bottom_left, 3)
        t_l (top_left, 1) <- m_m (center, 4)
        -------------------------------------
        b_r (bottom_right, 2) <- t_r (top_right, 0)
        b_r (bottom_right, 2) <- b_l (bottom_left, 3)
        b_r (bottom_right, 2) <- m_m (center, 4)
        -------------------------------------
        b_l (bottom_left, 3) <- t_l (top_left, 1)
        b_l (bottom_left, 3) <- b_r (bottom_right, 2)
        b_l (bottom_left, 3) <- m_m (center, 4)
        -------------------------------------
        m_m (center, 4) <- b_r (bottom_right, 2)
        m_m (center, 4) <- b_l (bottom_left, 3)
        m_m (center, 4) <- t_r (top_right, 0)
        m_m (center, 4) <- t_l (top_left, 1)
        -------------------------------------
     - The Object and the end-effector are connected:
        -------------------------------------
        e_e (end-effector) -> m_m (center, 4)
        e_e (end-effector) -> closest_corner_1
        e_e (end-effector) -> closest_corner_2
        -------------------------------------
        
        
    Arg:
        # obj_pos - Type: ndarray, 
                    Shape: (5,2), 
                    Definition: All corners and the centerpoint of the object, in the following order:
                    [["o_t_r_x", "o_t_r_y"], 
                     ["o_t_l_x", "o_t_l_y"], 
                     ["o_b_r_x", "o_b_r_y"], 
                     ["o_b_l_x", "o_b_l_y"], 
                     ["o_m_m_x", "o_m_m_y"]]
        # obj_vel - Type: ndarray, 
                    Shape: (5,2), 
                    Definition: All corners and the centerpoint of the object, in the following order:
                    [["o_t_r_x_v", "o_t_r_y_v"], 
                     ["o_t_l_x_v", "o_t_l_y_v"], 
                     ["o_b_r_x_v", "o_b_r_y_v"], 
                     ["o_b_l_x_v", "o_b_l_y_v"], 
                     ["o_m_m_x_v", "o_m_m_y_v"]]
        # tip_pos - Type: ndarray,
                    Shape: (2,),
                    Definition: The end effectors measured position: ["e_pos_x", "e_pos_y"]
        # tip_vel - Type: ndarray,
                    Shape: (2,),
                    Definition: The end effectors calculated velocity: ["e_vel_x", "e_vel_y"]
        # tip_con - Type: ndarray,
                    Shape: (1,),
                    Definition: Value is 1 if the object is in contact with the end effector, 0 otherwise.
        # do_cont - Type: Boolean
                    Definition: If True, then remove edges between the end effector and the object, keep edges otherwise.
                    
    #Returns:
        #dict - dictionary of the graph
        
    """
    
    
    # Nodes
    # Each node has an x,y position and an x_v,y_v velocity
    nodes = np.zeros((6, 5), dtype=np.float32)
    nodes[0, 0:2] = obj_pos[0]
    nodes[1, 0:2] = obj_pos[1]
    nodes[2, 0:2] = obj_pos[2]
    nodes[3, 0:2] = obj_pos[3]
    nodes[4, 0:2] = obj_pos[4]
    
    nodes[0, 2:4] = obj_vel[0]
    nodes[1, 2:4] = obj_vel[1]
    nodes[2, 2:4] = obj_vel[2]
    nodes[3, 2:4] = obj_vel[3]
    nodes[4, 2:4] = obj_vel[4]
    
    
    
    nodes[5, 0:2] = tip_pos
    nodes[5, 2:4] = tip_vel
    # 1 means that this node is the pusher node
    nodes[5, 4] = 1
    
    distances = np.sqrt(np.square(nodes[:3, 0] - nodes[5, 0]) + np.square(nodes[:3, 1] - nodes[5, 1]))
    tup_distances = list(set(zip([0, 1, 2, 3], distances)))

    def helper_second(elem):
        return elem[1]

    tup_distances.sort(key=helper_second)
    tup_distances = tup_distances[:2]
    
    

    # Edges
    obj_to_obj = 1.0
    ee_to_obj  = 0.0
    edge = np.array([[obj_to_obj, ee_to_obj]])
    edges = np.repeat(edge, 16, axis=0)
    senders = []
    receivers = []
    
    # Senders/Receivers
    # --------------- t_r -- t_l -- b_r -- b_l -- m_m-------
    senders.extend(  [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4,4])
    receivers.extend([1,2,4, 0,3,4, 0,3,4, 1,2,4, 0,1,2,3])
    
    if not do_cont:
        obj_to_obj = 0.0
        ee_to_obj  = 1.0
        senders.extend(  [5])
        receivers.extend([4])
        edge = np.array([[obj_to_obj, ee_to_obj]])
        edges = np.concatenate((edges, edge), axis=0)
        for node_numb, _ in tup_distances:
            senders.extend([5])  # fixed sends an edge
            receivers.extend([node_numb])  # not fixed receives it
            edge = np.array([[obj_to_obj, ee_to_obj]])
            edges = np.concatenate((edges, edge), axis=0)
        
    
    
    return {
        "globals": np.float32(tip_con),
        "nodes": np.float32(nodes),
        "edges": np.float32(edges),
        "receivers": receivers,
        "senders": senders}


def rigid_graph_from_pos_all(obj_pos, obj_vel, tip_pos, tip_vel, tip_con, do_cont = False):
    """
    Define dictionary containing the needed for building a 
    graph representation of the physical systems from the MIT push dataset.
    
    This function builds connectivity based on the following definition:
     - A rigid rectangle object consists of 5 nodes: The four corners of the object, and the center point of the object
     - The Object nodes are connected:
        -------------------------------------
        t_r (top_right, 0) <- t_l (top_left, 1)
        t_r (top_right, 0) <- b_r (bottom_right, 2)
        t_r (top_right, 0) <- m_m (center, 4)
        -------------------------------------
        t_l (top_left, 1) <- t_r (top_right, 0)
        t_l (top_left, 1) <- b_l (bottom_left, 3)
        t_l (top_left, 1) <- m_m (center, 4)
        -------------------------------------
        b_r (bottom_right, 2) <- t_r (top_right, 0)
        b_r (bottom_right, 2) <- b_l (bottom_left, 3)
        b_r (bottom_right, 2) <- m_m (center, 4)
        -------------------------------------
        b_l (bottom_left, 3) <- t_l (top_left, 1)
        b_l (bottom_left, 3) <- b_r (bottom_right, 2)
        b_l (bottom_left, 3) <- m_m (center, 4)
        -------------------------------------
        m_m (center, 4) <- b_r (bottom_right, 2)
        m_m (center, 4) <- b_l (bottom_left, 3)
        m_m (center, 4) <- t_r (top_right, 0)
        m_m (center, 4) <- t_l (top_left, 1)
        -------------------------------------
     - The Object and the end-effector are connected:
        -------------------------------------
        e_e (end-effector) -> m_m (center, 4)
        e_e (end-effector) -> b_l (bottom_left, 3)
        e_e (end-effector) -> b_r (bottom_right, 2)
        e_e (end-effector) -> b_r (top_left, 1)
        e_e (end-effector) -> b_r (top_right, 0)
        -------------------------------------
        
        
    Arg:
        # obj_pos - Type: ndarray, 
                    Shape: (5,2), 
                    Definition: All corners and the centerpoint of the object, in the following order:
                    [["o_t_r_x", "o_t_r_y"], 
                     ["o_t_l_x", "o_t_l_y"], 
                     ["o_b_r_x", "o_b_r_y"], 
                     ["o_b_l_x", "o_b_l_y"], 
                     ["o_m_m_x", "o_m_m_y"]]
        # obj_vel - Type: ndarray, 
                    Shape: (5,2), 
                    Definition: All corners and the centerpoint of the object, in the following order:
                    [["o_t_r_x_v", "o_t_r_y_v"], 
                     ["o_t_l_x_v", "o_t_l_y_v"], 
                     ["o_b_r_x_v", "o_b_r_y_v"], 
                     ["o_b_l_x_v", "o_b_l_y_v"], 
                     ["o_m_m_x_v", "o_m_m_y_v"]]
        # tip_pos - Type: ndarray,
                    Shape: (2,),
                    Definition: The end effectors measured position: ["e_pos_x", "e_pos_y"]
        # tip_vel - Type: ndarray,
                    Shape: (2,),
                    Definition: The end effectors calculated velocity: ["e_vel_x", "e_vel_y"]
        # tip_con - Type: ndarray,
                    Shape: (1,),
                    Definition: Value is 1 if the object is in contact with the end effector, 0 otherwise.
        # do_cont - Type: Boolean
                    Definition: If True, then remove edges between the end effector and the object, keep edges otherwise.
                    
    #Returns:
        #dict - dictionary of the graph
        
    """
    
    
    # Nodes
    # Each node has an x,y position and an x_v,y_v velocity
    nodes = np.zeros((6, 5), dtype=np.float32)
    nodes[0, 0:2] = obj_pos[0]
    nodes[1, 0:2] = obj_pos[1]
    nodes[2, 0:2] = obj_pos[2]
    nodes[3, 0:2] = obj_pos[3]
    nodes[4, 0:2] = obj_pos[4]
    
    nodes[0, 2:4] = obj_vel[0]
    nodes[1, 2:4] = obj_vel[1]
    nodes[2, 2:4] = obj_vel[2]
    nodes[3, 2:4] = obj_vel[3]
    nodes[4, 2:4] = obj_vel[4]
    
    nodes[5, 0:2] = tip_pos
    nodes[5, 2:4] = tip_vel
    # 1 means that this node is the pusher node
    nodes[5, 4] = 1
    
    distances = np.sqrt(np.square(nodes[:3, 0] - nodes[5, 0]) + np.square(nodes[:3, 1] - nodes[5, 1]))
    tup_distances = list(set(zip([0, 1, 2, 3], distances)))

    def helper_second(elem):
        return elem[1]

    tup_distances.sort(key=helper_second)
    tup_distances = tup_distances[:2]
    
    

    # Edges
    obj_to_obj = 1.0
    ee_to_obj  = 0.0
    edge = np.array([[obj_to_obj, ee_to_obj]])
    edges = np.repeat(edge, 16, axis=0)
    senders = []
    receivers = []
    
    # Senders/Receivers
    # --------------- t_r -- t_l -- b_r -- b_l -- m_m-------
    senders.extend(  [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4,4])
    receivers.extend([1,2,4, 0,3,4, 0,3,4, 1,2,4, 0,1,2,3])
    
    if not do_cont:
        obj_to_obj = 0.0
        ee_to_obj  = 1.0
        senders.extend([5])
        receivers.extend([4])
        edge = np.array([[obj_to_obj, ee_to_obj]])
        edges = np.concatenate((edges, edge), axis=0)
        for node_numb in range(5):
            senders.extend([5])  # fixed sends an edge
            receivers.extend([node_numb])  # not fixed receives it
            edge = np.array([[obj_to_obj, ee_to_obj]])
            edges = np.concatenate((edges, edge), axis=0)
        
    
    
    return {
        "globals": np.float32(tip_con),
        "nodes": np.float32(nodes),
        "edges": np.float32(edges),
        "receivers": receivers,
        "senders": senders}
    