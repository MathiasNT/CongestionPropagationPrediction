import torch
from pytorch_lightning.cli import LightningCLI

if __name__ =='__main__':
    from utils.data_utils.data_loader_utils import IncidentDataModule
    from models.baselines.lstm import LstmInformedModel

    