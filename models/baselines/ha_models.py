import torch
import numpy as np

class HA_model_old():
    """ Cannot deal with the hist data not large enough TODO discard when you have better
    """
    def __init__(self, train_dataset, emulated_dataset_size = None, random_seed=42):
        self.hist_incident_info = train_dataset.incident_info
        self.hist_ie_times = train_dataset.input_data_time[:,0,0,14,:]
        self.hist_targets = train_dataset.target_data

        if emulated_dataset_size is not None:
            np.random.seed(random_seed)
            full_idxs = np.arange(len(self.hist_incident_info))
            subset_idxs = np.random.choice(full_idxs, size=emulated_dataset_size, replace=False)

            self.hist_incident_info = self.hist_incident_info[subset_idxs]
            self.hist_ie_times = self.hist_ie_times[subset_idxs]
            self.hist_targets = self.hist_targets[subset_idxs]

    def find_best_hist_inci(self, cur_inc_info, cur_ie_time):
        cur_ie_idx = cur_inc_info[0].int()
        cur_n_lanes = cur_inc_info[1].int()

        edge_mask = (self.hist_incident_info[:,0] == cur_ie_idx)
        lane_mask = (self.hist_incident_info[:,1] == cur_n_lanes)
        combined_mask = edge_mask * lane_mask

        matching_ie_times = self.hist_ie_times[combined_mask]
        matching_idxs = combined_mask.argwhere()
        time_dists = torch.tensor([torch.linalg.vector_norm(cur_ie_time - ie_time) for ie_time in matching_ie_times])
        temp_index = time_dists.argmin()
        best_index = matching_idxs[temp_index]
        return best_index 

    def predict_dataset(self, cur_infos, cur_input_data, cur_input_data_time):
        cur_incident_time = cur_input_data_time[:,0,0,14,:]
        best_indexes = [self.find_best_hist_inci(info, time) for info, time in zip(cur_infos, cur_incident_time)]
        return self.hist_targets[best_indexes]

class HA_model_v1():
    """ Cannot deal with the hist data not large enough TODO discard when you have better
    """
    def __init__(self, train_dataset, emulated_dataset_size = None, random_seed=42):
        self.hist_incident_info = train_dataset.incident_info
        self.hist_net_info = train_dataset.network_info[:,:,0]
        self.hist_ie_times = train_dataset.input_time_data[:,0,0,-1,:]
        self.hist_targets = train_dataset.target_data

        if emulated_dataset_size is not None:
            np.random.seed(random_seed)
            full_idxs = np.arange(len(self.hist_incident_info))
            subset_idxs = np.random.choice(full_idxs, size=emulated_dataset_size, replace=False)

            self.hist_incident_info = self.hist_incident_info[subset_idxs]
            self.hist_ie_times = self.hist_ie_times[subset_idxs]
            self.hist_targets = self.hist_targets[subset_idxs]
            self.hist_net_info = self.hist_net_info[subset_idxs]

    def find_best_hist_inci(self, cur_inc_info, cur_ie_time):
        cur_ie_idx = cur_inc_info[0].int()
        cur_n_lanes = cur_inc_info[1].int()

        edge_cost = self.hist_net_info[:,cur_ie_idx.int()].abs()
        lane_cost = (self.hist_incident_info[:,1] - cur_n_lanes).abs()
        combined_cost = edge_cost + lane_cost
        min_cost = combined_cost.min()
        combined_mask = combined_cost == min_cost

        matching_ie_times = self.hist_ie_times[combined_mask]
        matching_idxs = combined_mask.argwhere()
        time_dists = torch.tensor([torch.linalg.vector_norm(cur_ie_time - ie_time) for ie_time in matching_ie_times])
        temp_index = time_dists.argmin()
        best_index = matching_idxs[temp_index]
        return best_index 

    def predict_dataset(self, cur_infos, cur_input_data, cur_input_data_time):
        cur_incident_time = cur_input_data_time[:,0,0,-1,:]
        best_indexes = [self.find_best_hist_inci(info, time) for info, time in zip(cur_infos, cur_incident_time)]
        return self.hist_targets[best_indexes]
    