import pandas as pd
import numpy as np
import sumolib
import json

from .network_utils import get_up_and_down_stream
from ..general_utils.conversion_utils import trig_transform


def transform_df_to_tensor(df, interpolation_lim=5, warmup = 10):
    """Takes the simulation dataframe and transforms it into a [E, L, T, F] np.array
    E: number of edges
    L: number of lanes (padded)
    T: Number of timesteps
    F: Number of features

    Args:
        df (pandas.DataFrame): Pandas DataFrame of the SUMO Detector XML output 
        interpolation_lim (int, optional): The limit of how many timesteps of missing speeds to interpolate. Defaults to 5.
        warmup (int, optional): How many timesteps to throw away for simulator warmup. Defaults to 10.

    Returns:
        data (np.array): array of shape [E, L, T, F]
        time (np.array): array of shape [T] with the timestamps
    """
    df = df.replace(-1, np.nan)
    df['edge_id'] = df.interval_id.str.split('_').str[1]
    df['lane_number'] = df.interval_id.str.split('_').str[-1]
    edges = df.edge_id.unique()
    lanes = df.lane_number.unique()
    time = df.interval_begin.unique()
    index_names = ['edge_id', 'lane_number', 'interval_begin']
    multi_index = pd.MultiIndex.from_product([edges, lanes, time], names = index_names)

    # Set the new index, resorting the data, padding -1 for missing lanes and then resetting the index s.t. 'interval_begin' is a column
    df = df.set_index(index_names)[['interval_occupancy', 'interval_speed', 'interval_nVehContrib']].reindex(multi_index, fill_value=-1).reset_index(level=2)
    
    if interpolation_lim != 0:    
        df = df.interpolate(limit=interpolation_lim, limit_area='inside')
        df = df.replace(np.nan, 0)
    
    #Make into np array of size [E, L, T, F] where F is 3 after time has been taken out
    data = df.values
    data = data.reshape(len(edges),len(lanes), len(time), -1)
    
    # Split the time and data into separate arrays
    time = data[0,0,warmup:,0]
    data = data[:,:,warmup:,1:]
    return data, time


def get_index_to_edge_dicts(df):
    """Creates dictionaries that go back and forth between edge ids and idxs

    Args:
        df (pd.DataFrame): dataframe from a Sumo simulation

    Returns:
        ind_to_edge (dict): dict from index to id
        edge_to_ind (dict): dict from id to index
    """
    df['edge_id'] = df.interval_id.str.split('_').str[1]
    edges = df.edge_id.unique()
    edge_to_ind = {edge:i for i,edge in enumerate(edges)}
    ind_to_edge = {i:edge for i,edge in enumerate(edges)}
    return ind_to_edge, edge_to_ind


def idxs_of_longest_seq(a1):
    """Returns the indexes and length of the longest consecutive sequence of True in the list

    Args:
        a1 (list): List of booleans

    Returns:
        idxs_longest_seq (np.array): Rows contain the start and end of the sequences of True's.  
        length_seq (np.array): Contains the length of each sequence of True's
    """
    # Pad False on both ends of array, find changes using np.diff, find idxs of changes
    idx_pairs = np.where(np.diff(np.hstack(([False],a1==1,[False]))))[0].reshape(-1,2)

    if len(idx_pairs) != 0:
        # Get the island lengths, whose argmax would give us the ID of longest island.
        # Start index of that island would be the desired output
        idxs_longest_seq = idx_pairs[np.diff(idx_pairs,axis=1).argmax()]
        length_seq = np.diff(idxs_longest_seq)[0]
    else:
        idxs_longest_seq = np.array([np.inf,-np.inf])
        length_seq = 0
    return idxs_longest_seq, length_seq


def find_congestion(residual_speed, lane_stds, length_lim, std_lim):
    """Takes in residual speed, lane stds and returns congestions that are longer than length_lim and more severe than std_lim times standard deviation

    Args:
        residual_speed (np.array): np.array of shape [T, E, L] with residual speeds.
        lane_stds (np.array): np.array of shape [E, L] with the lanes standard deviations
        length_lim (int): how many timesteps a congestion has to last to be considered 
        std_lim (float): how many standard deviations a congestions has to be worse than to be considered

    Returns:
        _type_: _description_
        affected_edges (np.array): boolean array of size [E] that indicates which edges are congested
        longest_affect_lengths
    """
    # Bool of affected lane at time steps [E, L ,T]
    affect_bool_arr = (residual_speed.transpose(2,0,1) < std_lim * -lane_stds)
    n_edges = residual_speed.shape[0]
    n_time_steps = residual_speed.shape[2]
    n_lanes = residual_speed.shape[1]

    # reshape for finding consecutive subsequences -> [E*L, T]
    arr = affect_bool_arr.transpose(1,2,0).reshape(-1, n_time_steps)

    # Find longest subsequences
    longest_affect_idxs, longest_affect_lengths = zip(*[idxs_of_longest_seq(row) for row in arr])

    # Reshape back
    longest_affect_lengths = np.stack(longest_affect_lengths).reshape(n_edges,n_lanes) 
    longest_affect_idxs = np.stack(longest_affect_idxs).reshape(n_edges,n_lanes,2)    
    affected_edges = (longest_affect_lengths > length_lim).any(axis=1)

    return affected_edges, longest_affect_lengths, longest_affect_idxs

def infer_incident_data(simulation_path):
    """Takes the detector data and incident information from a SUMO simulation at the given path and creates input and target data for congestion prediction.

    Args:
        simulation_path (str): path to the folder with the detectordata.csv and detectordata_counterfactual.csv files 
        scenario (str): which scenario the simulation is from. Used to find network file

    Returns:
        input_data (np.array): np.array w. shape [E, L, T_i, F] with input data for ML(T_i is time until incident)
        target_data (np.array): np.array w. shape [E,4] with target data for ML
        inci_data (np.array): np.array w. shape [E, L, T, F] with the full data from the incident simulation
        counter_data (np.array): np.array w. shape [E, L, T, F] with the full data from the counterfactual simulation
        ind_to_edge (dict): dict that goes from index to edge id
    """

    # Load incident information and data
    with open(f'{simulation_path}/incident_settings.json') as f:
        incident_settings = json.load(f)
    incident_edge = incident_settings['edge']

    inci_df = pd.read_csv(f'{simulation_path}/detectordata.csv', sep=';')
    counter_df = pd.read_csv(f'{simulation_path}/detectordata_counterfactual.csv', sep=';')
    
    min_inci_time = inci_df.interval_begin.min()
    max_inci_time = inci_df.interval_begin.max()
    min_counter_time = counter_df.interval_begin.min()
    max_counter_time = counter_df.interval_begin.max()
    min_time = np.max([min_inci_time, min_counter_time])
    max_time = np.min([max_inci_time, max_counter_time])
    counter_df = counter_df.loc[counter_df.interval_begin.between(min_time,max_time)]
    inci_df = inci_df.loc[inci_df.interval_begin.between(min_time,max_time)]   

    inter_lim = 2 # Has had best performance
    inci_data, time_sequence = transform_df_to_tensor(inci_df, interpolation_lim=inter_lim, warmup=10)
    counter_data, _ = transform_df_to_tensor(counter_df, interpolation_lim=inter_lim, warmup=10)

    n_edges = inci_data.shape[0]

    # Create masks 
    padded_lane_mask = ~(inci_data[...,1].mean(-1) == -1)
    unused_lanes_mask = ((counter_data[...,2].sum(2) > 0) & (counter_data[...,2].sum(2) < 300))
    inci_data[unused_lanes_mask] = -2
    counter_data[unused_lanes_mask] = -2
    
    residual_data = inci_data - counter_data

    # Load network information
    scenario_folder = simulation_path.split('Results')[0]
    net_path = f'{scenario_folder}/Simulations/Base/network.net.xml'
    net = sumolib.net.readNet(net_path)
    i_edge_obj = net.getEdge(incident_edge)
    _, upstream_edges, _, _= get_up_and_down_stream(i_edge_obj=i_edge_obj,
                                                    n_up=40,
                                                    n_down=40)    
    ind_to_edge, edge_to_ind = get_index_to_edge_dicts(inci_df)
    upstream_edge_idxs = [edge_to_ind[edge] for edge in upstream_edges]


    # Get normal condition lane STD
    lane_stds = counter_data[...,1].std(2)

    # Only look at speed
    residual_speed = residual_data[...,1]

    # Find the edges that are affected by congestions with harsher rules
    length_lim = 10
    (affected_edges, _, _) = find_congestion(residual_speed=residual_speed,
                                             lane_stds=lane_stds,
                                             length_lim=length_lim,
                                             std_lim=1.95)

    # Only look at the upstream congested edges
    upstream_edges_arr = np.isin(np.arange(n_edges), upstream_edge_idxs)
    affected_us_mask = (upstream_edges_arr * affected_edges)
    affected_upstream_edge_idxs = np.where(affected_us_mask)[0]

    # Find the start and end times for the edges that are congested using looser rules
    (_, longest_affect_lengths, longest_affect_idxs) = find_congestion(residual_speed=residual_speed,
                                                                                    lane_stds=lane_stds,
                                                                                    length_lim=1,
                                                                                    std_lim=0.5)

    cong_start_idxs = longest_affect_idxs[...,0]
    cong_end_idxs = longest_affect_idxs[...,1]

    cong_start_idxs[(longest_affect_lengths < length_lim)] = np.inf
    cong_end_idxs[(longest_affect_lengths < length_lim)] = -np.inf

    cong_start_time = cong_start_idxs.min(1)
    cong_end_time = cong_end_idxs.max(1)

    cong_start_time[~affected_us_mask] = 0
    cong_end_time[~affected_us_mask] = 0

    # Calculate the max speed decrease
    delta_speeds = np.zeros((n_edges))
    for edge in affected_upstream_edge_idxs:
        start_time = int(cong_start_time[edge])
        end_time = int(cong_end_time[edge])
        #TODO imlement the delta speed definition we want
        #counter_mean_speed = counter_data[edge, lane_mask[edge], start_time:end_time, 2].mean(-1)
        #inci_mean_speed = inci_data[edge, lane_mask[edge], start_time:end_time, 2].mean(-1)
        #delta_speeds[edge, lane_mask[edge]] = counter_mean_speed - inci_mean_speed
        delta_speeds[edge] = residual_speed[edge, padded_lane_mask[edge], start_time:end_time].min()
        
    # Do trig transform of time feature
    secs_in_day = 24 * 60 * 60
    trig_time = np.array([trig_transform(seconds, secs_in_day) for seconds in time_sequence])   
    trig_time = np.expand_dims(trig_time, axis=(0,1))
    trig_time = np.repeat(trig_time, inci_data.shape[0], axis=0)
    trig_time = np.repeat(trig_time, inci_data.shape[1], axis=1)
    inci_data = np.concatenate([inci_data, trig_time], axis=-1)
    
    # Create the input and target tensors
    incident_time = np.where(time_sequence == incident_settings['start_time'])[0].item()
    input_data = inci_data[:,:,:incident_time, :]
    #input_time = np.expand_dims(trig_time[:input_traffic_data.shape[2]], axis=(0,1))
    #input_time = np.repeat(input_time, input_traffic_data.shape[0], axis=0)
    #input_time = np.repeat(input_time, input_traffic_data.shape[1], axis=1)
    #input_data = np.concatenate([input_traffic_data, input_time], axis=-1)

    target_data = np.stack([affected_us_mask, cong_start_time, cong_end_time, delta_speeds], axis=-1)

    return input_data, target_data, inci_data, counter_data, ind_to_edge, incident_settings
