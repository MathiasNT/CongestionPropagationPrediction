from random import random
import yaml
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import datetime

from util_folder.ml_utils.data_utils.data_loader_utils import IncidentDataModule
from models.model_utils import load_configs, init_model


def test_config(config, overwrite_random_seed, overwrite_gpu):

    folder_path = f'Simulation_scenarios/{config["scenario"]}/Results/{config["simulation_name"]}'

    debug_run = config['debug']

    # Seed for reproducibility
    if overwrite_random_seed is not None: 
        random_seed = overwrite_random_seed
    else:
        random_seed = config['random_seed'] # For easy overwrite of seed
    pl.seed_everything(random_seed)

    # Overwrite gpu for easier scripting
    if overwrite_gpu is not None:
        gpu = overwrite_gpu
    else:
        gpu = config['gpu']

    # Load data    
    incident_data_module = IncidentDataModule(folder_path = folder_path, batch_size = config['batch_size'])
    if config['form'] == 'incident_only': # TODO could do asserts for other cases as well
        assert config['model'] in ['lstm', 'informed_lstm', 'mlp'], 'Only LSTM baselines run on incident only'

    # Init model
    model = init_model(config)

    # Init trainer
    if debug_run:
        trainer = pl.Trainer(max_epochs = config['epochs'],
                            accelerator="gpu",
                            devices=[gpu],
                            fast_dev_run=True,
                            auto_lr_find=config['infer_lr']
                            )
    else:
        now = datetime.datetime.now()
        time_str = f'{now.day}-{now.month}-{now.hour}{now.minute}'
        run_name = f"{config['wandb_name']}_{random_seed}_{time_str}"
        wandb_logger = WandbLogger(project=config['wandb_project']+'_test', name=run_name, save_dir='wandb_dir', log_model=True)
        #wandb.watch(model)
        wandb_logger.log_hyperparams(config)
        trainer = pl.Trainer(max_epochs = config['epochs'],
                            accelerator="gpu",
                            devices=[gpu], 
                            logger=wandb_logger,
                            auto_lr_find=config['infer_lr']
                            )

    # Test model
    trainer.test(model=model, datamodule=incident_data_module, ckpt_path=config['checkpoint_path'])
    wandb_logger.finalize(status='Succes')
    wandb.finish()

if __name__ == '__main__':
    # Load config yaml 
    parser = ArgumentParser()
    parser.add_argument('--config_paths', type=str, help='Path to the config YAML', nargs='+')
    parser.add_argument('--overwrite_random_seed', type=int, help='Overwrite the random seed from the configs')
    parser.add_argument('--overwrite_gpu', type=int, help='Overwrite the GPU from the config YAML')
    args = parser.parse_args()
    configs = []
    for config_path in args.config_paths:
        
        with open(config_path) as stream:
            config_dict = yaml.safe_load(stream)

        configs.append(load_configs(config_dict))
        

    for config in configs:
        test_config(config, args.overwrite_random_seed, args.overwrite_gpu)