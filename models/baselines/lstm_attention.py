import torch
from torch import nn

from ..base_model_class import BaseModelClass


class AttentionRNNModel(BaseModelClass):
    """Uninformed RNN baseline. 

    RNN model that sees each detector as a timeseries and independently tries to predict for each of them.
    The model gets no traffic information and predicts purely based on historic traffic data.

    OBS: Current implementation takes out the sensor on the edge with the incident.
    """
    def __init__(self, config, learning_rate):
        super().__init__(config, learning_rate)

        self.rnn = nn.LSTM(input_size = config['timeseries_in_size'],
                           hidden_size = config['rnn_hidden_size'], 
                           batch_first=True)

        self.attention = nn.MultiheadAttention(embed_dim=config['rnn_hidden_size'], 
                                               num_heads=config['attention_num_heads'], 
                                               batch_first=True)

        self.fc_shared = nn.Linear(in_features= config['rnn_hidden_size'], out_features=config['fc_hidden_size'])

        self.fc_classifier = nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_start = nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_end = nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_speed = nn.Linear(in_features=config['fc_hidden_size'], out_features=1)

    def forward(self, inputs, incident_info, network_info):
        batch_size, time_steps, features = inputs.shape

        output, (hn, _) = self.rnn(inputs)

        atn_emb = self.attention(output, output, output)[0] 
        
        hn = atn_emb[:,-1,:]

        hn = torch.relu(hn)

        hn_fc = self.fc_shared(hn)
        hn_fc = torch.relu(hn_fc)

        class_logit = self.fc_classifier(hn_fc)

        start_pred = self.fc_start(hn_fc)
        end_pred = self.fc_end(hn_fc)

        speed_pred = self.fc_speed(hn_fc)
    
        y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
        return y_hat

