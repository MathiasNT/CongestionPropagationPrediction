import torch
from torch import nn

from ..base_model_class import BaseModelClass


class MLPModel(BaseModelClass):
    """Uninformed MLP baseline

    MLP model that sees each detector as a vector input and independently tries to predict for each of them.
    """
    def __init__(self, config, learning_rate, pos_weights):
        super().__init__(config, learning_rate, pos_weights)
        self.relu = nn.ReLU() 
        self.mlp = nn.Sequential(
            nn.Linear(config['timeseries_in_size'] * config['n_timesteps'], config['mlp_hidden_size']),
            self.relu,
            nn.Linear(config['mlp_hidden_size'], config['mlp_hidden_size']),
            self.relu,
            nn.Linear(config['mlp_hidden_size'], config['mlp_hidden_size']),
            self.relu
        )


        self.fc_classifier = torch.nn.Linear(in_features=config['mlp_hidden_size'], out_features=1)
        self.fc_start = torch.nn.Linear(in_features=config['mlp_hidden_size'], out_features=1)
        self.fc_end = torch.nn.Linear(in_features=config['mlp_hidden_size'], out_features=1)
        self.fc_speed = torch.nn.Linear(in_features=config['mlp_hidden_size'], out_features=1)

    def forward(self, inputs, incident_info, network_info):
        batch_size, time_steps, features = inputs.shape

        inputs = inputs.reshape(batch_size, time_steps * features)

        h = self.mlp(inputs)

        class_logit = self.fc_classifier(h)

        start_pred = self.fc_start(h)
        end_pred = self.fc_end(h)

        speed_pred = self.fc_speed(h)
    
        y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
        return y_hat

