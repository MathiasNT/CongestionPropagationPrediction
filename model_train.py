from random import random
import yaml
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import datetime

from util_folder.ml_utils.data_utils.data_loader_utils import IncidentDataModule, RWIncidentDataModule
from models.model_utils import load_configs, init_model



def run_config(config, overwrite_random_seed, overwrite_gpu):


    debug_run = config['debug']

    # Seed for reproducibility
    if overwrite_random_seed is not None: 
        config['random_seed'] = overwrite_random_seed
    pl.seed_everything(config['random_seed'])

    # Overwrite gpu for easier scripting
    if overwrite_gpu is not None:
        config['gpu'] = overwrite_gpu

    # Load data 
    if config['scenario'] == 'pems':
        folder_path = f'{config["simulation_name"]}'
        incident_data_module = RWIncidentDataModule(folder_path = folder_path, 
                                                transform=config['transform'], 
                                                batch_size = config['batch_size'],
                                                spatial_test=config['spatial_test'],
                                                subset_size=config['subset_size'],
                                                min_impact_threshold=config['min_impact_threshold'])
    else:   
        folder_path = f'Simulation_scenarios/{config["scenario"]}/Results/{config["simulation_name"]}'
        incident_data_module = IncidentDataModule(folder_path = folder_path, 
                                                transform=config['transform'], 
                                                batch_size = config['batch_size'],
                                                spatial_test=config['spatial_test'],
                                                subset_size=config['subset_size'],
                                                min_impact_threshold=config['min_impact_threshold'])

    
    incident_data_module.setup()
    if config['form'] == 'incident_only': # TODO could do asserts for other cases as well
        assert config['model'] in ['lstm', 'informed_lstm', 'mlp'], 'Only LSTM baselines run on incident only'

    # Init model
    model = init_model(config=config, pos_weights=incident_data_module.pos_weights)

    # Init trainer
    if debug_run:
        trainer = pl.Trainer(max_epochs = config['epochs'],
                            accelerator="gpu",
                            devices=[config['gpu']],
                            fast_dev_run=True,
                            auto_lr_find=config['infer_lr'],
                            precision=config['lightning_precision']
                            )
    else:
        now = datetime.datetime.now()
        time_str = f'{now.day}-{now.month}-{now.hour}{now.minute}'
        run_name = f"{config['wandb_name']}_{config['random_seed']}_{time_str}"
        wandb_logger = WandbLogger(project=config['wandb_project'], name=run_name, save_dir='wandb_dir', log_model=True)
        #wandb.watch(model)
        wandb_logger.log_hyperparams(config)
        trainer = pl.Trainer(max_epochs = config['epochs'],
                            accelerator="gpu",
                            devices=[config['gpu']], 
                            logger=wandb_logger,
                            auto_lr_find=config['infer_lr'],
                            gradient_clip_val=0.5,
                            precision=config['lightning_precision']
                            )

    # Train model
    trainer.tune(model, datamodule=incident_data_module)
    
    if 'checkpoint_path' in config: # This is for loading a checkpoint
        trainer.fit(model=model,
                    datamodule=incident_data_module,
                    ckpt_path=config['checkpoint_path']
                    )
    else:
        trainer.fit(model=model,
                    datamodule=incident_data_module,
                    )
    
    if 'results_dir' in config:
        trainer.save_checkpoint(f'{config["results_dir"]}/checkpoint_{config["random_seed"]}.ckpt')

    # Test model
    if not debug_run:
        trainer.test(model=model, datamodule=incident_data_module)
        wandb_logger.finalize(status='Succes')
        wandb.finish()

if __name__ == '__main__':
    # Load config yaml 
    parser = ArgumentParser()
    parser.add_argument('--config_paths', type=str, help='Path to the config YAML', nargs='+')
    parser.add_argument('--overwrite_random_seed', type=int, help='Overwrite the random seed from the configs', nargs='+')
    parser.add_argument('--overwrite_gpu', type=int, help='Overwrite the GPU from the config YAML')
    args = parser.parse_args()
    configs = []
    for config_path in args.config_paths:
        
        with open(config_path) as stream:
            config_dict = yaml.safe_load(stream)

        configs.append(load_configs(config_dict))
        

    for config in configs:
        for seed in args.overwrite_random_seed:
            run_config(config, seed, args.overwrite_gpu)