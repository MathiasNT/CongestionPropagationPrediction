from importlib.resources import path
from locale import normalize
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pytorch_lightning as pl
import numpy as np


class IncidentDataSet(Dataset):
    """Torch DataSet class that contains traffic data, inferred congestion backlogs and incident information

    Attributes:
        input_data (torch.Tensor): [N_s, E, L, T_i, F] (Occupancy, Speed, NvehContrib, trig_time1, trig_time2) with traffic data before the incident
        target_data (torch.Tensor): [N_s, E,4] with target data for ML (affected_bool, start_time, end_time, delta speed)
        incident_info (torch.Tensor): [N_s, 4] with incident info (incident edge, number of blocked lanes, slow zone speed, block duration)
        network_info (torch.Tensor): [N_s, E, 13] with network information (0: upstream/downstream level, 1-7: unused lanes mask, 8-13: padded lane mask)
    """

    def __init__(self, input_data, target_data, incident_info, network_info):
        self.input_data_obs = input_data[...,:-2]
        self.input_data_time = input_data[...,-2:]
        self.target_data = target_data
        self.incident_info = incident_info
        self.network_info = network_info


    def __len__(self):
        return len(self.input_data_obs)
    
    def __getitem__(self, idx):
        input_obs_batch = self.input_data_obs[idx]
        input_time_batch = self.input_data_time[idx]
        target_batch = self.target_data[idx]
        incident_info_batch = self.incident_info[idx]
        network_info_batch = self.network_info[idx]

        return {'input': input_obs_batch,
                'time': input_time_batch,
                'target': target_batch,
                'incident_info': incident_info_batch,
                'network_info': network_info_batch}

# TODO clean up the code for the trig time features. They can be split out earlier
class IncidentDataModule(pl.LightningDataModule):
    """pl DataModule that creates and keeps track of the IncidentDataSet and DataLoaders

        Attributes:
            batch_size (int): Batch size for the dataloaders
            folder_path (str): Path to the folder with the data
            train_frac (float): Fraction of data to be used for the train set, the rest is split 50/50 to val and test sets

            Datasets for the Dataloader  - see IncidentDataSet for specifics 
    """

    def __init__(self, folder_path, train_frac=0.6,  batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.folder_path = folder_path
        self.train_frac = train_frac 

    def prepare_data(self):
        # TODO implement check of the folder here -> have code in notebbok on data preprocess
        return
    
    def setup(self, stage=None):
        input_full = np.load(f'{self.folder_path}/input_data.npy')
        target_full = np.load(f'{self.folder_path}/target_data.npy')
        incident_info_full = np.load(f'{self.folder_path}/incident_info.npy')
        network_info_full = np.load(f'{self.folder_path}/network_info.npy')

        input_full = torch.Tensor(input_full)
        target_full = torch.Tensor(target_full)
        incident_info_full = torch.Tensor(incident_info_full)
        network_info_full = torch.Tensor(network_info_full)

        padded_lane_mask = torch.BoolTensor(network_info_full[:,:,7:] > 0)

        train_len = int(np.ceil(len(input_full) * self.train_frac))
        test_val_len = int(np.round((len(input_full) - train_len) * 0.5))
        train_set, val_set, test_set = random_split(range(len(input_full)), [train_len, test_val_len,test_val_len])

        normalized_input, self.input_obs_mean, self.input_obs_std = self.normalize_input(train_set, input_full, padded_lane_mask)

        self.input_train = normalized_input[train_set.indices]
        self.input_val = normalized_input[val_set.indices]
        self.input_test = normalized_input[test_set.indices]

        self.target_train = target_full[train_set.indices]
        self.target_val = target_full[val_set.indices]
        self.target_test = target_full[test_set.indices]

        self.incident_info_train = incident_info_full[train_set.indices]
        self.incident_info_val = incident_info_full[val_set.indices]
        self.incident_info_test = incident_info_full[test_set.indices]

        self.network_info_train = network_info_full[train_set.indices]
        self.network_info_val = network_info_full[val_set.indices]
        self.network_info_test = network_info_full[test_set.indices]

    def train_dataloader(self):
        train_split = IncidentDataSet(self.input_train,
                                      self.target_train,
                                      self.incident_info_train,
                                      self.network_info_train)
        return DataLoader(train_split, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        val_split = IncidentDataSet(self.input_val,
                                      self.target_val,
                                      self.incident_info_val,
                                      self.network_info_val)
        return DataLoader(val_split, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        test_split = IncidentDataSet(self.input_test,
                                      self.target_test,
                                      self.incident_info_test,
                                      self.network_info_test)
        return DataLoader(test_split, batch_size=self.batch_size, num_workers=8)

    @staticmethod
    def normalize_input(train_set, input_full, padded_lane_mask):
        # Get mean and std based on train set
        train_input = input_full[train_set.indices]
        masked_input = train_input[padded_lane_mask[train_set.indices]]
        input_obs_mean = masked_input[..., :3].mean((0,1))
        input_obs_std = masked_input[..., :3].std((0,1))

        # Normalize the full data
        normalized_input_obs = (input_full[..., :3] - input_obs_mean) / input_obs_std

        normalized_input = torch.ones_like(input_full) * -1
        normalized_input[..., :3] = normalized_input_obs
        normalized_input[..., 3:] = input_full[..., 3:]

        normalized_input[~padded_lane_mask] = -1

        return normalized_input, input_obs_mean, input_obs_std