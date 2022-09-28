'''
Quick n dirty script for debugging
'''
import yaml
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch
import numpy as np

from utils.data_utils.data_loader_utils import IncidentDataModule
from models.baselines.lstm import RnnInformedModel, RnnModel, RnnNetworkInformedModel
from models.my_graph.mpnn import MLPDecoder
from models.model_utils import load_configs, create_gnn_args
from models.rose_models.lgf_model import SimpleGNN, InformedGNN

if __name__ == '__main__':
    # Load config yaml 
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to the config YAML', required=True)
    args = parser.parse_args()
    with open(args.config_path) as stream:
        config_dict = yaml.safe_load(stream)

    config = load_configs(config_dict) 


    folder_path = f'{config["scenario"]}/Results/{config["simulation_name"]}'


    # Seed for reproducibility
    pl.seed_everything(config['random_seed'])


    # Load data    
    incident_data_module = IncidentDataModule(folder_path = folder_path, batch_size = config['batch_size'])
    if config['form'] == 'incident_only':
        assert config['model'] in ['lstm', 'informed_lstm'], 'Only LSTM baselines run on incident only'

    # Init model
    if config['model'] == 'lstm':
        model = RnnModel(config, learning_rate=config['learning_rate'])

    elif config['model'] == 'informed_lstm':
        model = RnnInformedModel(config, learning_rate=config['learning_rate'])
    
    elif config['model'] == 'network_informed_lstm':
        model = RnnNetworkInformedModel(config, learning_rate=config['learning_rate'])

    elif config['model'] == 'mpnn_gcn':
        adj_mx =  torch.Tensor(np.load(config['AD_path']))
        model = MLPDecoder(adj_mx=adj_mx, config=config, learning_rate=config['learning_rate'])

    elif config['model'] == 'gnn':
        gnn_args = create_gnn_args(config)
        adj_mx =  torch.Tensor(np.load(config['AD_path']))
        model = SimpleGNN(adj_mx=adj_mx, args=gnn_args, config=config, learning_rate=1e-3) 

    elif config['model'] == 'informed_gnn':
        gnn_args = create_gnn_args(config)
        adj_mx =  torch.Tensor(np.load(config['AD_path']))
        model = InformedGNN(adj_mx=adj_mx, args=gnn_args, config=config, learning_rate=1e-3) 


    # Init trainer
    if config['debug']:
        trainer = pl.Trainer(max_epochs = config['epochs'],
                            accelerator="gpu",
                            devices=[config['gpu']],
                            fast_dev_run=True,
                            auto_lr_find=config['infer_lr']
                            )
    else:
        wandb_logger = WandbLogger(project=config['wandb_project'], name=f"{config['wandb_name']}-{config['random_seed']}", save_dir='wandb_dir', log_model=True)
        #wandb.watch(model)
        wandb_logger.log_hyperparams(config)
        trainer = pl.Trainer(max_epochs = config['epochs'],
                            accelerator="gpu",
                            devices=[config['gpu']], 
                            logger=wandb_logger,
                            auto_lr_find=config['infer_lr']
                            )

    # Train model
   # trainer.tune(model, datamodule=incident_data_module)
    #trainer.fit(model=model,
                #datamodule=incident_data_module,
                #)
    
    # Test model
    trainer.test(model=model,
                 datamodule=incident_data_module)