'''
Quick n dirty script for debugging
'''
from pyexpat import model
import pytorch_lightning as pl
from utils.data_utils.data_loader_utils import IncidentDataModule
from models.baselines.lstm import RnnInformedModel, RnnModel, RnnNetworkInformedModel
from pytorch_lightning.loggers import WandbLogger
import yaml
from pathlib import Path
from argparse import ArgumentParser


if __name__ == '__main__':
    # Load config yaml 
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to the config YAML', required=True)
    args = parser.parse_args()
    with open(args.config_path) as stream:
        config = yaml.safe_load(stream)

    folder_path = f'{config["scenario"]}/Results/{config["simulation_name"]}'

    # Load data    
    incident_data_module = IncidentDataModule(folder_path = folder_path, batch_size = 2048)
    if config['incident_only']:
        assert config['model'] in ['lstm', 'informed_lstm'], 'Only LSTM baselines run on incident only'

    # Init model
    if config['model'] == 'lstm':
        model = RnnModel(config, learning_rate=1e-3)
    elif config['model'] == 'informed_lstm':
        model = RnnInformedModel(config, learning_rate=1e-3)
    elif config['model'] == 'network_informed_lstm':
        model = RnnNetworkInformedModel(config, learning_rate=1e-3)

    # Init trainer
    if config['debug']:
        trainer = pl.Trainer(max_epochs = config['epochs'],
                            accelerator="gpu",
                            devices=[config['gpu']],
                            fast_dev_run=True,
                            auto_lr_find=config['infer_lr']
                            )
    else:
        wandb = WandbLogger(project=config['wandb_project'], name=config['wandb_name'], log_model=True)
        #wandb.watch(model)
        trainer = pl.Trainer(max_epochs = config['epochs'],
                            accelerator="gpu",
                            devices=[config['gpu']], 
                            logger=wandb,
                            auto_lr_find=config['infer_lr']
                            )

    # Train model
    trainer.tune(model, datamodule=incident_data_module)
    trainer.fit(model=model,
                datamodule=incident_data_module,
                )
    
    # Test model
    trainer.test(model=model,
                 datamodule=incident_data_module)