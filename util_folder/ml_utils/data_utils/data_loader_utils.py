from importlib.resources import path
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pytorch_lightning as pl
import numpy as np



class Standardize:
    """Note that the Standardize only standardizes the time series input""" 
    def __call__(self, input_obs_full, target_full, incident_info_full, network_info_full, train_set, padded_lane_mask, unused_lane_mask):
        # Get mean and std based on train set
        train_input = input_obs_full[train_set.indices]
        masked_input = train_input[~padded_lane_mask[train_set.indices]]
        input_obs_mean = masked_input.mean((0,1))
        input_obs_std = masked_input.std((0,1))

        # Normalize the full data
        standardized_input_obs = (input_obs_full - input_obs_mean) / input_obs_std

        # Reapply mask
        standardized_input_obs[padded_lane_mask] = -1

        self.params = {'mean': input_obs_mean, 'std':input_obs_std}

        return standardized_input_obs, target_full, incident_info_full, network_info_full


class ScaleNormalize:
    def __call__(self, input_obs_full, target_full, incident_info_full, network_info_full, train_set, padded_lane_mask, unused_lane_mask):
        # Normalize input
        train_input = input_obs_full[train_set.indices]
        lane_mask = padded_lane_mask + unused_lane_mask
        masked_input = train_input[~lane_mask[train_set.indices]]
        input_min = masked_input.view(-1, masked_input.shape[-1]).min(0).values
        input_max = masked_input.view(-1, masked_input.shape[-1]).max(0).values
        scaled_input_obs = (input_obs_full - input_min) / (input_max - input_min) 
        input_param_dict = {'min': input_min, 'max': input_max} 

        # Normalize target TODO think about if the affected mask would help?
        target_full_reg = target_full[...,1:]
        target_full_reg[...,-1] *= -1
        train_target_reg = target_full_reg[train_set.indices]        
        target_min = train_target_reg.view(-1, train_target_reg.shape[-1]).min(0).values
        target_max = train_target_reg.view(-1, train_target_reg.shape[-1]).max(0).values
        scaled_target_full_reg = (target_full_reg - target_min) / (target_max - target_min) 
        scaled_target_full_reg[...,-1] *= -1
        target_full[...,1:] = scaled_target_full_reg
        scaled_target_full = target_full
        target_param_dict = {'min': target_min, 'max': target_max}
        
        # Normalize incident_info
        incident_info_full_edge = incident_info_full[...,1:]        
        train_incident_info_edge = incident_info_full_edge[train_set.indices]        
        incident_info_min = train_incident_info_edge.view(-1, train_incident_info_edge.shape[-1]).min(0).values
        incident_info_min[1] = 0 # TODO fix hotfix by adding noise to data
        incident_info_max = train_incident_info_edge.view(-1, train_incident_info_edge.shape[-1]).max(0).values
        scaled_incident_info_full_edge = (incident_info_full_edge - incident_info_min) / (incident_info_max - incident_info_min)
        incident_info_full[...,1:] = scaled_incident_info_full_edge
        scaled_incident_info_full = incident_info_full
        incident_info_param_dict = {'min': incident_info_min, 'max': incident_info_max}

        # Scale network info - used for indexing so no normalize here. 
        scaled_network_info_full = network_info_full

        self.params = {'input': input_param_dict, 'target': target_param_dict, 'incident_info': incident_info_param_dict}

        return scaled_input_obs, scaled_target_full, scaled_incident_info_full, scaled_network_info_full


class IncidentDataSet(Dataset):
    """Torch DataSet class that contains traffic data, inferred congestion backlogs and incident information

    Attributes:
        input_data (torch.Tensor): [N_s, E, L, T_i, F] (Occupancy, Speed, NvehContrib with traffic data before the incident
        input_data_time (torch.Tensor): [N_s, E, L, T_i, F_time] (trig_time1, trig_time2) * Note this is in the input data during init
        target_data (torch.Tensor): [N_s, E,4] with target data for ML (affected_bool, start_time, end_time, delta speed)
        incident_info (torch.Tensor): [N_s, 4] with incident info (incident edge, number of blocked lanes, slow zone speed, block duration)
        network_info (torch.Tensor): [N_s, E, 13] with network information (0: upstream/downstream level, 1-7: unused lanes mask, 8-13: padded lane mask)
    """

    def __init__(self, input_obs_data, input_time_data, target_data, incident_info, network_info):
        self.input_obs_data = input_obs_data
        self.input_time_data = input_time_data
        self.target_data = target_data
        self.incident_info = incident_info
        self.network_info = network_info


    def __len__(self):
        return len(self.input_obs_data)
    
    def __getitem__(self, idx):
        input_obs_batch = self.input_obs_data[idx]
        input_time_batch = self.input_time_data[idx]
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

                Datasets for the Dataloader  - see IncidentDataSet for specifics 
            train_frac (float): Fraction of data to be used for the train set, the rest is split 50/50 to val and test sets
            input_val/test/train (torch.Tensor): The input datasets for the val, test and train dataloaders
            target_val/test/train (torch.Tensor): The target datasets for the val, test and train dataloaders
            incident_info_val/test/train (torch.Tensor): The incident_info datasets for the val, test and train dataloaders
            network_info_val/test/train (torch.Tensor): The network_infor datasets for the val, test and train dataloaders

            full_pos weight (float): reweighting factor based on full network for the positive class for the BCE loss 
                                    (# negative obs / # positive obs)
            incident_edge_pos weight (float): reweighting factor based only on incident edge for the positive class for the 
                                             BCE loss (# negative obs / # positive obs)

    """

    def __init__(self, folder_path, transform, train_frac=0.6,  batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.folder_path = folder_path
        self.train_frac = train_frac

        if transform == 'standardize':
            self.transform = Standardize()
        elif transform == 'scalenormalize':
            self.transform = ScaleNormalize()
        

    def prepare_data(self):
        # TODO implement check of the folder here -> have code in notebbok on data preprocess
        return
    
    def setup(self, stage=None):
        input_full = torch.Tensor(np.load(f'{self.folder_path}/input_data.npy'))
        input_obs_full = input_full[...,:-2]
        input_time_full = input_full[...,-2:]
        target_full = torch.Tensor(np.load(f'{self.folder_path}/target_data.npy'))
        incident_info_full = torch.Tensor(np.load(f'{self.folder_path}/incident_info.npy'))
        network_info_full = torch.Tensor(np.load(f'{self.folder_path}/network_info.npy'))

        n_obs_full = len(input_full)
        
        print(f'*** DATA SUMMARY: ***')
        print(f'{input_obs_full.shape=}')
        print(f'{input_time_full.shape=}')
        print(f'{target_full.shape=}')
        print(f'{incident_info_full.shape=}')
        print(f'{network_info_full.shape=}\n')


        # Generate the train, val and test split idxs
        train_len = int(np.ceil(n_obs_full * self.train_frac))
        test_val_len = int(np.round((n_obs_full - train_len) * 0.5))
        train_set, val_set, test_set = random_split(range(n_obs_full),
                                                    [train_len, test_val_len,test_val_len],
                                                    generator=torch.Generator().manual_seed(1))


        # Normalize the data
        padded_lane_mask = torch.BoolTensor(network_info_full[:,:,7:] == 0)
        unused_lane_mask = torch.BoolTensor(network_info_full[:,:,1:7] > 0)
        tfed_input_obs, tfed_target_full, tfed_incident_info_full, tfed_network_info_full = self.transform(input_obs_full=input_obs_full, 
                                                                                                           target_full=target_full, 
                                                                                                           incident_info_full=incident_info_full, 
                                                                                                           network_info_full=network_info_full,
                                                                                                           train_set=train_set,
                                                                                                           padded_lane_mask=padded_lane_mask,
                                                                                                           unused_lane_mask=unused_lane_mask
                                                                                                           )
        input_time_full[padded_lane_mask] = -1 # TODO clean up the path of time

        # Split up the normalized data
        self.input_obs_train = tfed_input_obs[train_set.indices]
        self.input_obs_val = tfed_input_obs[val_set.indices]
        self.input_obs_test = tfed_input_obs[test_set.indices]

        self.input_time_train = input_time_full[train_set.indices]
        self.input_time_val = input_time_full[val_set.indices]
        self.input_time_test = input_time_full[test_set.indices]

        self.target_train = tfed_target_full[train_set.indices]
        self.target_val = tfed_target_full[val_set.indices]
        self.target_test = tfed_target_full[test_set.indices]

        self.incident_info_train = tfed_incident_info_full[train_set.indices]
        self.incident_info_val = tfed_incident_info_full[val_set.indices]
        self.incident_info_test = tfed_incident_info_full[test_set.indices]

        self.network_info_train = tfed_network_info_full[train_set.indices]
        self.network_info_val = tfed_network_info_full[val_set.indices]
        self.network_info_test = tfed_network_info_full[test_set.indices]


        pos_weights = []
        target_class = self.target_train[...,0]
        for i in range(0,30):
            mask = self.network_info_train[...,0] == -i
            pos_weight = (target_class[mask] == 0).sum() / target_class[mask].sum()
            pos_weights.append(pos_weight)
        self.pos_weights = torch.tensor(pos_weights)

        # This part should be moved to preprocessing
        # But for now we calculate the positive weight of the classification problem
        self.full_pos_weight = (self.target_train[...,0] == 0).sum() / self.target_train[...,0].sum()
        incident_edge_target = self.target_train[(self.network_info_train[...,0] == 0)]
        self.incident_edge_pos_weight =  (incident_edge_target[...,0] == 0).sum() / (incident_edge_target[...,0]).sum()

    def train_dataloader(self):
        train_split = IncidentDataSet(input_obs_data=self.input_obs_train, 
                                      input_time_data=self.input_time_train,
                                      target_data=self.target_train,
                                      incident_info=self.incident_info_train,
                                      network_info=self.network_info_train)
        return DataLoader(train_split, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        val_split = IncidentDataSet(input_obs_data=self.input_obs_val, 
                                      input_time_data=self.input_time_val,
                                      target_data=self.target_val,
                                      incident_info=self.incident_info_val,
                                      network_info=self.network_info_val)
        return DataLoader(val_split, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        test_split = IncidentDataSet(input_obs_data=self.input_obs_test, 
                                      input_time_data=self.input_time_test,
                                      target_data=self.target_test,
                                      incident_info=self.incident_info_test,
                                      network_info=self.network_info_test)
        return DataLoader(test_split, batch_size=self.batch_size, shuffle=False, num_workers=8)

