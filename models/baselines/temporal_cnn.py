import torch
from torch import dropout, nn
from ..base_model_class import BaseModelClass


# TODO add weight initialization to match Temporal CNN paper
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super().__init__()

        self.relu = nn.ReLU()

        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0)) # Might be tricky with the batch dim?

        #TODO Add weightnorm and dropout to this for it to match the temporal cnn paper probably
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, (kernel_size), stride=stride, padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, (kernel_size), stride=stride, padding=0, dilation=dilation)

        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.pad, self.conv2, self.relu)


        # This is used the residual link if input and output dim is not the same
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)



class TemporalCNNModel(BaseModelClass):
    """Uninformed RNN baseline. 

    RNN model that sees each detector as a timeseries and independently tries to predict for each of them.
    The model gets no traffic information and predicts purely based on historic traffic data.

    OBS: Current implementation takes out the sensor on the edge with the incident.
    """
    def __init__(self, config, learning_rate):
        super().__init__(config, learning_rate)

        layers = []

        num_levels = len(config['num_channels'])

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = config['cnn_in_size'] if i == 0 else config['num_channels'][i - 1]
            out_channels = config['num_channels'][i]
            layers += [TemporalBlock(n_inputs=in_channels,
                                     n_outputs=out_channels, 
                                     kernel_size=config['kernel_size'],
                                     stride=1,
                                     dilation=dilation_size,
                                     padding=(config['kernel_size'] - 1) * dilation_size)]

        self.net = nn.Sequential(*layers)

        self.fc_shared = torch.nn.Linear(in_features= config['tcn_emb_size'], out_features=config['fc_hidden_size'])

        self.fc_classifier = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_start = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_end = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_speed = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)



    def forward(self, inputs, incident_info, network_info):

        batch_size, time_steps, features = inputs.shape

        inputs = inputs.permute(0,2,1)

        tcn_emb = self.net(inputs)
        tcn_emb = tcn_emb.reshape(batch_size, -1)

        hn = torch.relu(tcn_emb)

        hn_fc = self.fc_shared(hn)
        hn_fc = torch.relu(hn_fc)

        class_logit = self.fc_classifier(hn_fc)

        start_pred = self.fc_start(hn_fc)
        end_pred = self.fc_end(hn_fc)

        speed_pred = self.fc_speed(hn_fc)
    
        y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
        return y_hat
