import pytorch_lightning as pl
import torch


class LstmModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size = config['lstm_input_size'],
                                    hidden_size = config['lstm_hidden_size'],
                                    batch_first=True)

        self.fc_shared = torch.nn.Linear(in_features= config['lstm_hidden_size'], out_features=config['fc_hidden_size'])

        self.fc_classifier = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_start = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_end = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_speed = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)

        self.bce_loss_func = torch.nn.BCELoss()
        self.mse_loss_func = torch.nn.MSELoss()

    def forward(self, x):
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

    def get_combined_loss(self, y_hat, y):
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
        y_true = batch['target']
        batch_size, n_nodes, n_lanes, n_timesteps, n_features = x.shape

        x = x.permute(0,1,2,4,3)
        x = x.reshape(batch_size * n_nodes,n_lanes * n_features, n_timesteps)
        x = x.permute(0,2,1)

        y_hat = self.forward(x)
        y_hat = y_hat.reshape(batch_size, n_nodes, -1)

        bce_loss, start_loss, end_loss, speed_loss = self.get_combined_loss(y_hat, y_true)
        loss = bce_loss + start_loss + end_loss + speed_loss 

        accuracy = self.get_accuracy(y_hat[...,0], y_true[...,0])

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['input']
        y_true = batch['target']
        batch_size, n_nodes, n_lanes, n_timesteps, n_features = x.shape

        x = x.permute(0,1,2,4,3)
        x = x.reshape(batch_size * n_nodes,n_lanes * n_features, n_timesteps)
        x = x.permute(0,2,1)

        y_hat = self.forward(x)
        y_hat = y_true.reshape(batch_size, n_nodes, -1)

        bce_loss, start_loss, end_loss, speed_loss = self.get_combined_loss(y_hat, y_true)
        loss = bce_loss + start_loss + end_loss + speed_loss 
        accuracy = self.get_accuracy(y_hat[...,0], y_true[...,0])

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_bce_loss', bce_loss, on_epoch=True)
        self.log('val_start_loss', start_loss, on_epoch=True)
        self.log('val_end_loss', end_loss, on_epoch=True)
        self.log('val_speed_loss', speed_loss, on_epoch=True)
        self.log('val_accuracy', accuracy, on_epoch=True)
    
        #return {'val_loss': loss,
                #'val_bce_loss': bce_loss,
                #'val_start_loss': start_loss,
                #'val_end_loss': end_loss,
                #'val_speed_loss': speed_loss,
                #'val_accuracy': accuracy}

    # TODO test the new log to see if this is even necessary
    #def validation_epoch_end(self, outputs):
        #avg_loss = torch.stack([losses['val_loss'] for losses in outputs]).mean()
        #avg_bce_loss = torch.stack([losses['val_bce_loss'] for losses in outputs]).mean()
        #avg_start_loss = torch.stack([losses['val_start_loss'] for losses in outputs]).mean()
        #avg_end_loss = torch.stack([losses['val_end_loss'] for losses in outputs]).mean()
        #avg_speed_loss = torch.stack([losses['val_speed_loss'] for losses in outputs]).mean()
        #avg_accuracy = torch.stack([losses['val_accuracy'] for losses in outputs]).mean()

        #self.log('val_loss', avg_loss)
        #self.log('val_bce_loss', avg_bce_loss)
        #self.log('val_start_loss', avg_start_loss)
        #self.log('val_end_loss', avg_end_loss)
        #self.log('val_speed_loss', avg_speed_loss)
        #self.log('val_accuracy', avg_accuracy)


    def test_step(self, batch, batch_idx):
        x = batch['input']
        y_true = batch['target']
        batch_size, n_nodes, n_lanes, n_timesteps, n_features = x.shape

        x = x.permute(0,1,2,4,3)
        x = x.reshape(batch_size * n_nodes,n_lanes * n_features, n_timesteps)
        x = x.permute(0,2,1)

        y_hat = self.forward(x)
        y_hat = y_true.reshape(batch_size, n_nodes, -1)

        bce_loss, start_loss, end_loss, speed_loss = self.get_combined_loss(y_hat, y_true)
        loss = bce_loss + start_loss + end_loss + speed_loss 
        accuracy = self.get_accuracy(y_hat[...,0], y_true[...,0])

        self.log('test_loss', loss, on_epoch=True)
        self.log('test_bce_loss', bce_loss, on_epoch=True)
        self.log('test_start_loss', start_loss, on_epoch=True)
        self.log('test_end_loss', end_loss, on_epoch=True)
        self.log('test_speed_loss', speed_loss, on_epoch=True)
        self.log('test_accuracy', accuracy, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)