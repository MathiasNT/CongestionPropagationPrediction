import pytorch_lightning as pl
import torch


# Train a full model on all of the data as well
# Get started on visualizing the predictions somehow
# Write Rose
# Go through code and clean it up if you have nothing else to do.

class LstmModelBase(pl.LightningModule):
    """Lightning module base for LSTM based models.

    Main difference from main base model is that this reshapes each sensor into the batch dimension.
    I.e. This removes any spatial information.
    """

    def __init__(self, config):
        super().__init__()
        self.incident_only = config['incident_only']

        self.bce_loss_func = torch.nn.BCELoss()
        self.mse_loss_func = torch.nn.MSELoss()
        
        self.save_hyperparameters()

    def get_combined_loss(self, y_hat, y):
        assert y_hat.shape == y.shape, "Shapes do not match"
        bce_loss = self.bce_loss_func(y_hat[...,0], y[...,0])
        start_loss = self.mse_loss_func(y_hat[...,1], y[...,1])
        end_loss = self.mse_loss_func(y_hat[...,2], y[...,2])
        speed_loss = self.mse_loss_func(y_hat[...,3], y[...,3])
        return bce_loss, start_loss, end_loss, speed_loss

    @staticmethod
    def get_accuracy(y_hat, y_true):
        y_hat = y_hat.flatten()
        y_true = y_true.flatten()
        assert y_true.ndim == 1 and y_true.size() == y_hat.size()
        y_hat = y_hat > 0.5
        return (y_true == y_hat).sum().item() / y_true.size(0)

    def training_step(self, batch, batch_idx):
        x = batch['input']
        time = batch['time']
        y_true = batch['target']
        incident_info = batch['incident_info']
        network_info = batch['network_info']

        batch_size = x.shape[0]
        batch_incident_mask = (network_info[:,:, 0] == 0)

        x, incident_info = self.reshape_inputs(x, time, incident_info, batch_incident_mask)

        y_hat = self.forward(x, incident_info)

        y_hat, y_true = self.reshape_targets(y_hat, y_true, batch_incident_mask, batch_size)

        bce_loss, start_loss, end_loss, speed_loss = self.get_combined_loss(y_hat, y_true)
        loss = bce_loss + start_loss + end_loss + speed_loss 
        accuracy = self.get_accuracy(y_hat[...,0], y_true[...,0])

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/bce_loss', bce_loss, on_step=False, on_epoch=True)
        self.log('train/start_loss', start_loss, on_step=False, on_epoch=True)
        self.log('train/end_loss', end_loss, on_step=False, on_epoch=True)
        self.log('train/speed_loss', speed_loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['input']
        time = batch['time']
        y_true = batch['target']
        incident_info = batch['incident_info']
        network_info = batch['network_info']

        batch_size = x.shape[0]
        batch_incident_mask = (network_info[:,:, 0] == 0)
        
        x, incident_info = self.reshape_inputs(x, time, incident_info, batch_incident_mask)

        y_hat = self.forward(x, incident_info)

        y_hat, y_true = self.reshape_targets(y_hat, y_true, batch_incident_mask, batch_size)

        bce_loss, start_loss, end_loss, speed_loss = self.get_combined_loss(y_hat, y_true)
        loss = bce_loss + start_loss + end_loss + speed_loss 
        accuracy = self.get_accuracy(y_hat[...,0], y_true[...,0])

        self.log('val/loss', loss, on_epoch=True)
        self.log('val/bce_loss', bce_loss, on_epoch=True)
        self.log('val/start_loss', start_loss, on_epoch=True)
        self.log('val/end_loss', end_loss, on_epoch=True)
        self.log('val/speed_loss', speed_loss, on_epoch=True)
        self.log('val/accuracy', accuracy, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # TODO double check if this function is necessary, I think PL will just use the val function
        x = batch['input']
        time = batch['time']
        y_true = batch['target']
        incident_info = batch['incident_info']
        network_info = batch['network_info']

        batch_size = x.shape[0]
        batch_incident_mask = (network_info[:,:, 0] == 0)

        x, incident_info = self.reshape_inputs(x, time, incident_info, batch_incident_mask)

        y_hat = self.forward(x, incident_info)

        y_hat, y_true = self.reshape_targets(y_hat, y_true, batch_incident_mask, batch_size)

        bce_loss, start_loss, end_loss, speed_loss = self.get_combined_loss(y_hat, y_true)
        loss = bce_loss + start_loss + end_loss + speed_loss 
        accuracy = self.get_accuracy(y_hat[...,0], y_true[...,0])

        self.log('test/loss', loss, on_epoch=True)
        self.log('test/bce_loss', bce_loss, on_epoch=True)
        self.log('test/start_loss', start_loss, on_epoch=True)
        self.log('test/end_loss', end_loss, on_epoch=True)
        self.log('test/speed_loss', speed_loss, on_epoch=True)
        self.log('test/accuracy', accuracy, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def reshape_inputs(self, x, time, incident_info, batch_incident_mask):
        batch_size, n_nodes, n_lanes, n_timesteps, n_obs_features = x.shape
        n_time_features = time.shape[-1]

        if self.incident_only:
            x = x[batch_incident_mask]
            x = x.permute(0,2,1,3)
            x = x.reshape(batch_size, n_timesteps, n_lanes * n_obs_features)
            time = time[:,0,0,:,:]
            x = torch.cat([x, time], dim=-1)
        else:
            x = x.permute(0,1,3,2,4)            
            x = x.reshape(batch_size, n_nodes, n_timesteps, n_lanes * n_obs_features)
            
            time = time[:,:,0,:,:]
            x = torch.cat([x, time], dim=-1)
            
            x = x.reshape(-1, n_timesteps, n_lanes * n_obs_features + n_time_features)
            
            incident_info = incident_info.unsqueeze(1).expand(-1, n_nodes, -1)
            incident_info = incident_info.reshape(-1, incident_info.shape[-1] )

        return x, incident_info

    def reshape_targets(self, y_hat, y_true, batch_incident_mask, batch_size):
        
        if self.incident_only:
            y_true = y_true[batch_incident_mask]
            y_hat = y_hat.reshape(batch_size, -1)
        else:
            y_hat = y_hat.reshape(y_true.shape)


        return y_hat, y_true

class LstmModel(LstmModelBase):
    """Uninformed LSTM baseline. 

    LSTM model that sees each detector as a timeseries and independently tries to predict for each of them.
    The model gets no traffic information and predicts purely based on historic traffic data.

    OBS: Current implementation takes out the sensor on the edge with the incident.
    """
    def __init__(self, config):
        super().__init__(config)

        self.lstm = torch.nn.LSTM(input_size = config['lstm_input_size'],
                                    hidden_size = config['lstm_hidden_size'],
                                    batch_first=True)

        self.fc_shared = torch.nn.Linear(in_features= config['lstm_hidden_size'], out_features=config['fc_hidden_size'])

        self.fc_classifier = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_start = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_end = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_speed = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)

    def forward(self, x, incident_info):
        batch_size, time_steps, features = x.shape

        _, (hn, _) = self.lstm(x)
        hn = torch.relu(hn)

        hn_fc = self.fc_shared(hn)
        hn_fc = torch.relu(hn_fc)

        class_logit = self.fc_classifier(hn_fc)
        class_logit = torch.sigmoid(class_logit)

        start_pred = self.fc_start(hn_fc)
        end_pred = self.fc_end(hn_fc)

        speed_pred = self.fc_speed(hn_fc)
    
        y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
        return y_hat


class LstmInformedModel(LstmModelBase):
    """Informed LSTM baseline.

    Extension of the uninformed lstm baseline. This LSTM gets incident information but still treats each sensor as an independent time series.


    OBS: Current implementation only looks at the sensor closest to the incident.
    """
    def __init__(self, config):
        super().__init__(config)

        self.lstm = torch.nn.LSTM(input_size = config['lstm_input_size'],
                                    hidden_size = config['lstm_hidden_size'],
                                    batch_first=True)

        self.fc_shared = torch.nn.Linear(in_features= config['lstm_hidden_size'], out_features=config['fc_hidden_size'])

        self.fc_info = torch.nn.Linear(in_features=config['info_size'], out_features=config['fc_hidden_size'] )

        self.fc_inform = torch.nn.Linear(in_features=config['fc_hidden_size'] * 2, out_features=config['fc_hidden_size'])

        self.fc_classifier = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_start = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_end = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_speed = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)

    def forward(self, x, incident_info):
        batch_size, time_steps, features = x.shape

        _, (hn, _) = self.lstm(x)
        hn = torch.relu(hn)

        hn_fc = self.fc_shared(hn)
        hn_fc = torch.relu(hn_fc).squeeze()

        info = incident_info[:,1:]   # selecting number of blocked lanes, slow zone speed and duration and startime 
        info_embed = self.fc_info(info) 
        info_embed = torch.relu(info_embed)

        # need to replicate the info along the batch dim here

        hn_fc_informed = self.fc_inform(torch.cat([hn_fc, info_embed], dim=-1))
        hn_fc_informed = torch.relu(hn_fc_informed)

        class_logit = self.fc_classifier(hn_fc_informed)
        class_logit = torch.sigmoid(class_logit)

        start_pred = self.fc_start(hn_fc_informed)
        end_pred = self.fc_end(hn_fc_informed)

        speed_pred = self.fc_speed(hn_fc_informed)
    
        y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
        return y_hat