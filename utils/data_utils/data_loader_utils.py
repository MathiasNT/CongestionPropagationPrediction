from importlib.resources import path
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pytorch_lightning as pl
import numpy as np

class IncidentDataSet(Dataset):
    def __init__(self, input_data, target_data, incident_settings):
        self.input_data = input_data
        self.target_data = target_data
        self.incident_settings = incident_settings
        
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        input_batch = self.input_data[idx]
        target_batch = self.target_data[idx]
        incident_settings_batch = self.incident_settings[idx]
        
        return {'input': input_batch,
                'target': target_batch}
                #'incident_settings': incident_settings_batch} TODO Get rid of the dictionary here it messes up the dataloader


class IncidentDataModule(pl.LightningDataModule):
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
        incident_settings_full = np.load(f'{self.folder_path}/incident_settings.npy', allow_pickle=True)

        input_full = torch.Tensor(input_full)
        target_full = torch.Tensor(target_full)


        train_len = int(np.ceil(len(input_full) * self.train_frac))
        test_val_len = int(np.round((len(input_full) - train_len) * 0.5))
        train_set, val_set, test_set = random_split(range(len(input_full)), [train_len, test_val_len,test_val_len])

        self.input_train = input_full[train_set.indices]
        self.input_val = input_full[val_set.indices]
        self.input_test = input_full[test_set.indices]

        self.target_train = target_full[train_set.indices]
        self.target_val = target_full[val_set.indices]
        self.target_test = target_full[test_set.indices]

        self.incident_settings_train = incident_settings_full[train_set.indices]
        self.incident_settings_val = incident_settings_full[val_set.indices]
        self.incident_settings_test = incident_settings_full[test_set.indices]

    def train_dataloader(self):
        train_split = IncidentDataSet(self.input_train,
                                      self.target_train,
                                      self.incident_settings_train)
        return DataLoader(train_split, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        val_split = IncidentDataSet(self.input_val,
                                      self.target_val,
                                      self.incident_settings_val)
        return DataLoader(val_split, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        test_split = IncidentDataSet(self.input_test,
                                      self.target_test,
                                      self.incident_settings_test)
        return DataLoader(test_split, batch_size=self.batch_size, num_workers=8)
