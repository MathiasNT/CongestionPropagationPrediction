def flatten(l):
    """flattens a list of lists of varying lengths

    Args:
        l (list): list of lists of varying lengths

    Returns:
        list: flattened list w. same contents as l
    """
    return [item for sublist in l for item in sublist]

def get_upstream_edges(edge_obj, n_up):
    """Creates a dict of upstream edges.

    Args:
        edge_obj (sumolib.edge.Edge): sumolib Edge of the edge that the upstream should be created from
        n_up (int): How many levels of upstream to put in dict

    Returns:
        dict: dict of upstream edges. Keys are upstream levels and values are edge ids.
    """
    upstream_edges = {}
    upstream_edges[0] = list(edge_obj.getIncoming().keys())

    for i in range(0, n_up):
        next_edges = []
        for edge in upstream_edges[i]:
            next_edges += (list(edge.getIncoming().keys()))
        upstream_edges[i+1] = next_edges
        
    upstream_edges_ids = {}
    for i in range(0, n_up + 1):
        level_ids = []
        for edge in upstream_edges[i]:
            level_ids += [edge.getID()]
        upstream_edges_ids[i] = level_ids

    return upstream_edges_ids 

def get_downstream_edges(edge_obj, n_down):
    """Creates a dict of downstream edges.

    Args:
        edge_obj (sumolib.edge.Edge): sumolib Edge of the edge that the downstream should be created from
        n_down (int): How many levels of downstream to put in dict

    Returns:
        dict: dict of downstream edges. Keys are downstream levels and values are edge ids.
    """
    downstream_edges = {}
    downstream_edges[0] = list(edge_obj.getOutgoing().keys())

    for i in range(0, n_down):
        next_edges = []
        for edge in downstream_edges[i]:
            next_edges += (list(edge.getOutgoing().keys()))
        downstream_edges[i+1] = next_edges
        
    downstream_edges_ids = {}
    for i in range(0, n_down + 1):
        level_ids = []
        for edge in downstream_edges[i]:
            level_ids += [edge.getID()]
        downstream_edges_ids[i] = level_ids
    return downstream_edges_ids        

def get_edge_to_level_dict_string(upstream_edges_ids, downstream_edges_ids, incident_edge):
    """Creates a dict that goes from edge to level information string.
        For plotting.
    Args:
        upstream_edges_ids (dict): upstream level dict
        downstream_edges_ids (dict): downstream level dict
        incident_edge (str): id of the incident edge 

    Returns:
        dict: dict w. edge ids as keys and corresponding str w. upstream or downstream level as value
    """
    edge_to_level_dict = {}
    for level in upstream_edges_ids.keys():
        for i, edge in enumerate(upstream_edges_ids[level]):
            edge_to_level_dict[edge] = f'upstream_{level}_{i}'

    for level in downstream_edges_ids.keys():
        for i, edge in enumerate(downstream_edges_ids[level]):
            edge_to_level_dict[edge] = f'downstream_{level}_{i}'

    edge_to_level_dict[incident_edge] = 'incident_edge'

    return edge_to_level_dict
     
def get_edge_to_level_dict_numerical(upstream_edges_ids, downstream_edges_ids, incident_edge):
    """Creates a dict that goes from edge to level information in numerical format.
        Positive means downstream, negative means upstream
        TODO what to do with disconnected nodes? For now I will give them a very high value
        Intuition is that being disconnected has same effect as being very far downstream
    Args:
        upstream_edges_ids (dict): upstream level dict
        downstream_edges_ids (dict): downstream level dict
        incident_edge (str): id of the incident edge 

    Returns:
        dict: dict w. edge ids as keys and corresponding numerical upstream or downstream level as value
    """
    edge_to_level_dict = {}
    for level in upstream_edges_ids.keys():
        for edge in upstream_edges_ids[level]:
            edge_to_level_dict[edge] = -(level + 1)

    for level in downstream_edges_ids.keys():
        for edge in downstream_edges_ids[level]:
            edge_to_level_dict[edge] = (level + 1)

    edge_to_level_dict[incident_edge] = 0

    return edge_to_level_dict
    
def get_up_and_down_stream(i_edge_obj, n_up, n_down):
    """Function to get the upstream edges, downstream edges and edge_level_dict of the given incident.

    Args:
        i_edge_obj sumolib.edge.Edge: sumolib Edge object of the incident edge
        n_up (int): How many levels of upstream to do 
        n_down (int): How many levels of downstream to do

    Returns:
        edge_to_level_dict (dict): Dict that goes from edge id to upstrean or downstream level
        upstream_edges (list of str): List of upstream and incident edge ids
        downstream_edges (list of str): List of downstream and incident edge ids
        relevant_edges (list of str): List of downstream, upstream and incident edge ids
    """
    incident_edge = i_edge_obj.getID()
    downstream_edges_ids = get_downstream_edges(i_edge_obj, n_up)
    all_downstream_edges_ids = flatten(list(downstream_edges_ids.values()))

    upstream_edges_ids = get_upstream_edges(i_edge_obj, n_down)
    all_upstream_edges_ids = flatten(list(upstream_edges_ids.values()))


    edge_to_level_dict = get_edge_to_level_dict_numerical(upstream_edges_ids, downstream_edges_ids, incident_edge)

    relevant_edges = all_upstream_edges_ids + all_downstream_edges_ids + [incident_edge]
    upstream_edges = all_upstream_edges_ids + [incident_edge]
    downstream_edges = all_downstream_edges_ids + [incident_edge]

    return edge_to_level_dict, upstream_edges, downstream_edges, relevant_edges