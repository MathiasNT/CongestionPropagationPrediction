'''
Quick n dirty script for debugging
'''
from pyexpat import model
import pytorch_lightning as pl
from utils.data_utils.data_loader_utils import IncidentDataModule
from models.baselines.lstm import LstmInformedModel, LstmModel
from pytorch_lightning.loggers import WandbLogger


if __name__ == '__main__':
    scenario = 'motorway'
    experiment_name = 'incident3'
    folder_path = f'{scenario}/Results/{experiment_name}'

    config = {'model': 'informed_lstm',
             'incident_only': True,
             'lstm_input_size': 20, 
             'lstm_hidden_size': 32, 
             'fc_hidden_size': 16, 
             'info_size': 3,
             'epochs': 2000,
             'wand_logger': False}


    if config['model'] == 'lstm':
        model = LstmModel(config)
    elif config['model'] == 'informed_lstm':
        model = LstmInformedModel(config)

    if config['wand_logger']:
        wandb = WandbLogger(project='incident')
        trainer = pl.Trainer(max_epochs = config['epochs'],
                            accelerator="gpu",
                            devices=1, 
                            logger=wandb)
    else:
        trainer = pl.Trainer(max_epochs = config['epochs'],
                            accelerator="gpu",
                            devices=1)
 
    incident_data_module = IncidentDataModule(folder_path = folder_path, batch_size = 2048)

    trainer.fit(model=model,
                datamodule=incident_data_module,
                )
    trainer.test(model=model,
                 datamodule=incident_data_module)