import torch
from torch import nn

from ..base_model_class import BaseModelClass


class AttentionRNNModel(BaseModelClass):
    """Uninformed RNN baseline with self attention on LSTM hiddenstates. 
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


class InformedAttentionRNNModel(BaseModelClass):
    """Informed RNN baseline. 

    Extension of the uninformed attention baseline. This model gets incident information but still treats each sensor as an independent time series.

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

        self.fc_info = torch.nn.Linear(in_features=config['info_in_size'], out_features=config['fc_hidden_size'] )
        self.fc_inform = torch.nn.Linear(in_features=config['fc_hidden_size'] * 2, out_features=config['fc_hidden_size'])

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

        info = incident_info[:,1:]   # selecting number of blocked lanes, slow zone speed and duration and startime 
        info_embed = self.fc_info(info) 
        info_embed = torch.relu(info_embed)
        # need to replicate the info along the batch dim here
        hn_fc_informed = self.fc_inform(torch.cat([hn_fc, info_embed], dim=-1))
        hn_fc_informed = torch.relu(hn_fc_informed)

        class_logit = self.fc_classifier(hn_fc_informed)

        start_pred = self.fc_start(hn_fc_informed)
        end_pred = self.fc_end(hn_fc_informed)

        speed_pred = self.fc_speed(hn_fc_informed)
    
        y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
        return y_hat


class NetworkInformedAttentionRNNModel(BaseModelClass):
    """Informed RNN baseline. 

    Extension of the uninformed attention baseline. This model gets incident information but still treats each sensor as an independent time series.

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

        self.fc_info = torch.nn.Linear(in_features=config['info_in_size'] + config['network_in_size'], out_features=config['fc_hidden_size'] )
        self.fc_inform = torch.nn.Linear(in_features=config['fc_hidden_size'] * 2, out_features=config['fc_hidden_size'])

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

        info = incident_info[:,1:]   # selecting number of blocked lanes, slow zone speed and duration and startime 
        combined_info = torch.cat([info,network_info.unsqueeze(-1)], dim=-1)       
        combined_info_embed = self.fc_info(combined_info) 
        combined_info_embed = torch.relu(combined_info_embed)
        # need to replicate the info along the batch dim here
        hn_fc_informed = self.fc_inform(torch.cat([hn_fc, combined_info_embed], dim=-1))
        hn_fc_informed = torch.relu(hn_fc_informed)

        class_logit = self.fc_classifier(hn_fc_informed)

        start_pred = self.fc_start(hn_fc_informed)
        end_pred = self.fc_end(hn_fc_informed)

        speed_pred = self.fc_speed(hn_fc_informed)
    
        y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
        return y_hat