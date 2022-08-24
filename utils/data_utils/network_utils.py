def flatten(l):
    return [item for sublist in l for item in sublist]

def get_upstream_edges(edge_obj, n_up):
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

def get_edge_to_level_dict(upstream_edges_ids, downstream_edges_ids, incident_edge):
    edge_to_level_dict = {}
    for level in upstream_edges_ids.keys():
        for i, edge in enumerate(upstream_edges_ids[level]):
            edge_to_level_dict[edge] = f'upstream_{level}_{i}'

    for level in downstream_edges_ids.keys():
        for i, edge in enumerate(downstream_edges_ids[level]):
            edge_to_level_dict[edge] = f'downstream_{level}_{i}'

    edge_to_level_dict[incident_edge] = 'incident_edge'

    return edge_to_level_dict    
    