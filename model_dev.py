import pytorch_lightning as pl
from utils.data_utils.data_loader_utils import IncidentDataModule
from models.baselines.lstm import LstmModel

if __name__ == '__main__':
    scenario = 'motorway'
    experiment_name = 'incident3'
    folder_path = f'{scenario}/Results/{experiment_name}'

    incident_data_module = IncidentDataModule(folder_path = folder_path, batch_size = 1024)
    #incident_data_module.setup(path, 0.6)
    #train_dataloader = incident_data_module.train_dataloader()
    #i, batch = next(enumerate(train_dataloader))

    config = {'lstm_input_size': 30, 'lstm_hidden_size': 32, 'fc_hidden_size': 16, }


    lstm_model = LstmModel(config)

    #lstm_model.training_step(batch, i)

    trainer = pl.Trainer(max_epochs = 100, accelerator="gpu", devices=1)
    trainer.fit(model=lstm_model,
                datamodule=incident_data_module,
                )