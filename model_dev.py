'''
Quick n dirty script for debugging
'''
from pyexpat import model
import pytorch_lightning as pl
from utils.data_utils.data_loader_utils import IncidentDataModule
from models.baselines.lstm import LstmInformedModel, LstmModel, LstmNetworkInformedModel
from pytorch_lightning.loggers import WandbLogger
import yaml
from pathlib import Path
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to the config YAML')
    args = parser.parse_args()

    with open(args.config_path) as stream:
        config = yaml.safe_load(stream)

    scenario = config['scenario']
    simulation_name = config['simulation_name']
    folder_path = f'{scenario}/Results/{simulation_name}'

    if config['incident_only']:
        assert config['model'] in ['lstm', 'informed_lstm'], 'Only LSTM baselines run on incident only'

    if config['model'] == 'lstm':
        model = LstmModel(config)
    elif config['model'] == 'informed_lstm':
        model = LstmInformedModel(config)
    elif config['model'] == 'network_informed_lstm':
        model = LstmNetworkInformedModel(config)


    if config['debug']:
        trainer = pl.Trainer(max_epochs = config['epochs'],
                            accelerator="gpu",
                            devices=1,
                            fast_dev_run=True
                            )
    else:
        wandb = WandbLogger(project='incident', log_model=True)
        trainer = pl.Trainer(max_epochs = config['epochs'],
                            accelerator="gpu",
                            devices=1, 
                            logger=wandb)
    incident_data_module = IncidentDataModule(folder_path = folder_path, batch_size = 2048)

    trainer.fit(model=model,
                datamodule=incident_data_module,
                )
    trainer.test(model=model,
                 datamodule=incident_data_module)