import pandas as pd
import numpy as np

def transform_df_to_tensor(df):
    df = df.replace(-1, np.nan)
    df['edge_id'] = df.interval_id.str.split('_').str[1]
    df['lane_number'] = df.interval_id.str.split('_').str[-1]
    edges = df.edge_id.unique()
    lanes = df.lane_number.unique()
    time = df.interval_begin.unique()
    index_names = ['edge_id', 'lane_number', 'interval_begin']
    multi_index = pd.MultiIndex.from_product([edges, lanes, time], names = index_names)
    df = df.set_index(index_names)[['interval_occupancy', 'interval_speed']].reindex(multi_index, fill_value=-1).reset_index(level=2)
    df = df.interpolate() # TODO think about if this is a good idea again
    data = df.values
    data = data.reshape(len(edges),len(lanes), len(time), -1)

    return data

def get_index_to_edge_dicts(df):
    df['edge_id'] = df.interval_id.str.split('_').str[1]
    edges = df.edge_id.unique()
    edge_to_ind = {edge:i for i,edge in enumerate(edges)}
    ind_to_edge = {i:edge for i,edge in enumerate(edges)}
    return ind_to_edge, edge_to_ind