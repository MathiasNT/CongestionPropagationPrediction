
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import MeanAbsolutePercentageError, Accuracy
from torchmetrics.classification import BinaryF1Score
from torchmetrics.functional import precision_recall

from util_folder.ml_utils.result_utils.metric_utils import masked_mape
from util_folder.ml_utils.loss_utils import UpstreamBCELoss
        
            
class BaseModelClass(pl.LightningModule):
    """Lightning module base for RNN based models.

    Main difference from main base model is that this reshapes each sensor into the batch dimension.
    I.e. This removes any spatial information.
    """

    def __init__(self, config, learning_rate, pos_weights=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.form = config['form']
        assert self.form in ['incident_only', 'independent', 'graph'], 'Please select prober data form' # TODO test
        

        self.loss_type = config['loss_type']
        self.bce_pos_weight = torch.Tensor([config['bce_pos_weight']])

        self.bce_loss_func = nn.BCEWithLogitsLoss(pos_weight=self.bce_pos_weight)
        self.mse_loss_func = nn.MSELoss()
        self.mape_loss_func = MeanAbsolutePercentageError()
        self.mae_loss_func = nn.L1Loss()
        self.acc_func = Accuracy()
        self.f1_func = BinaryF1Score()

        if self.loss_type == 'upstream_loss':
            self.upstream_bce_loss_func = UpstreamBCELoss(pos_weights=pos_weights)

        self.seed = config['random_seed']

        self.transform_network_info_bool = config['transform_network_info']

        if 'results_dir' in config.keys():
            self.results_dir = config['results_dir']
        else:
            self.results_dir = None


    def training_step(self, batch, batch_idx):
        loss, _, _ = self.standard_step(batch=batch, step_type='train')
        return loss

    def validation_step(self, batch, batch_idx):
        self.standard_step(batch=batch, step_type='val')

    def test_step(self, batch, batch_idx):
        _, y_hat_batch, y_true_batch = self.standard_step(batch=batch, step_type='test')
        return torch.stack([y_hat_batch, y_true_batch])

    def test_epoch_end(self, output_results):
        output_results_full = torch.cat(output_results, dim=1)
        y_hat = output_results_full[0]
        y_true = output_results_full[1]
        torch.save(y_hat, f'{self.logger.experiment.dir}/y_hat.pt')
        torch.save(y_true, f'{self.logger.experiment.dir}/y_true.pt')
        
        if self.results_dir is not None:
            torch.save(y_hat, f'{self.results_dir}/y_hat_{self.seed}.pt')
            torch.save(y_true, f'{self.results_dir}/y_true_{self.seed}.pt')

    def standard_step(self, batch, step_type):
        x = batch['input']
        time = batch['time']
        y_true = batch['target']
        incident_info = batch['incident_info']
        network_info = batch['network_info'].clone()

        batch_size = x.shape[0]
        batch_incident_mask = (network_info[:,:, 0] == 0) # Done before transform of network info so allways the same

        x, incident_info, network_info = self.reshape_inputs(x, time, incident_info, network_info, batch_incident_mask)

        # TODO make sure the transform of the network info works for al transforms 
        # (IE_only, parrallel and graph)
        if self.transform_network_info_bool:
            network_info = self.transform_network_info(network_info=network_info)

        y_hat = self.forward(inputs=x,
                             incident_info=incident_info,
                             network_info=network_info)

        y_hat, y_true = self.reshape_targets(y_hat, y_true, batch_incident_mask, batch_size)


        bce_loss, start_loss, end_loss, speed_loss = self.calculate_losses(y_hat, y_true)

        if self.loss_type == 'full':
            loss = bce_loss + start_loss + end_loss + speed_loss 
        elif self.loss_type == 'upstream_loss':
            upstream_bce_loss = self.upstream_bce_loss_func(y_hat[...,0], y_true[...,0], batch['network_info'][...,0])
            loss = upstream_bce_loss + start_loss + end_loss + speed_loss
            self.log(f'{step_type}/upstream_bce_loss', upstream_bce_loss, on_step=False,  on_epoch=True)
        elif self.loss_type == 'bce_only':
            loss = bce_loss        

        metrics_dict = self.calc_metrics(y_hat, y_true, step_type)

        self.log(f'{step_type}/loss', loss, on_step=False, on_epoch=True)
        self.log(f'{step_type}/bce_loss', bce_loss, on_step=False,  on_epoch=True)
        self.log(f'{step_type}/start_loss', start_loss, on_step=False,  on_epoch=True)
        self.log(f'{step_type}/end_loss', end_loss, on_step=False,  on_epoch=True)
        self.log(f'{step_type}/speed_loss', speed_loss, on_step=False,  on_epoch=True)
        self.log_dict(metrics_dict, on_step=False, on_epoch=True)
        #self.log(f'{step_type}/accuracy', accuracy, on_step=False,  on_epoch=True) TODO clean up
        #self.log(f'{step_type}/precision', precision, on_step=False,  on_epoch=True)
        #self.log(f'{step_type}/recall', recall, on_step=False,  on_epoch=True)
        #self.log(f'{step_type}/f1', f1, on_step=False,  on_epoch=True)
        return loss, y_hat.detach(), y_true.detach()

    def calculate_losses(self, y_hat, y):
        assert y_hat.shape == y.shape, "Shapes do not match"
        bce_loss = self.bce_loss_func(y_hat[...,0], y[...,0])
        start_loss = self.mse_loss_func(y_hat[...,1], y[...,1])
        end_loss = self.mse_loss_func(y_hat[...,2], y[...,2])
        speed_loss = self.mse_loss_func(y_hat[...,3], y[...,3])
        return bce_loss, start_loss, end_loss, speed_loss

    def calc_metrics(self, y_hat, y, step_type):
        assert y_hat.shape == y.shape, "Shapes do not match"

        metric_dict = {}

        metric_dict[f'{step_type}/accuracy'] = self.acc_func(y_hat[...,0], y[...,0].int())
        metric_dict[f'{step_type}/f1'] = self.f1_func(y_hat[...,0], y[...,0].int())
        metric_dict[f'{step_type}/prec'], metric_dict[f'{step_type}/rec'] = precision_recall(y_hat[...,0], y[...,0].int())

        # Masked MAPE
        metric_dict[f'{step_type}/start_Mmape'] = masked_mape(y_hat[...,1], y[...,1])
        metric_dict[f'{step_type}/end_Mmape'] = masked_mape(y_hat[...,2], y[...,2])
        metric_dict[f'{step_type}/speed_Mmape'] = masked_mape(y_hat[...,3], y[...,3])

        return metric_dict
        
    def reshape_inputs(self, x, time, incident_info, network_info, batch_incident_mask):
        batch_size, n_nodes, n_lanes, n_timesteps, n_obs_features = x.shape
        n_time_features = time.shape[-1]

        if self.form == 'incident_only':
            x = x[batch_incident_mask]
            x = x.permute(0,2,1,3)
            x = x.reshape(batch_size, n_timesteps, n_lanes * n_obs_features)
            time = time[:,0,0,:,:]
            x = torch.cat([x, time], dim=-1)

        elif self.form == 'graph':
            x = x.permute(0,1,3,2,4)
            x = x.reshape(batch_size, n_nodes, n_timesteps, n_lanes * n_obs_features)
            time = time[:,:,0,:,:]
            x = torch.cat([x, time], dim=-1)
            # TODO figure out what to do with the network_info and incident info
    
        elif self.form == 'independent':
            x = x.permute(0,1,3,2,4)            
            x = x.reshape(batch_size, n_nodes, n_timesteps, n_lanes * n_obs_features)
            time = time[:,:,0,:,:]
            x = torch.cat([x, time], dim=-1)
            x = x.reshape(-1, n_timesteps, n_lanes * n_obs_features + n_time_features)
            incident_info = incident_info.unsqueeze(1).expand(-1, n_nodes, -1)
            incident_info = incident_info.reshape(-1, incident_info.shape[-1] )
            network_info = network_info[...,0].reshape(-1)

        return x, incident_info, network_info
    

    def transform_network_info(self, network_info):
        inci_net_info = network_info[...,0]
        local_upstream_mask = inci_net_info.le(0) & inci_net_info.ge(-30)
        inci_net_info[local_upstream_mask] = torch.exp(0.25*inci_net_info[local_upstream_mask])
        inci_net_info[~local_upstream_mask] = 0
        network_info[...,0] = inci_net_info
        return network_info


    def reshape_targets(self, y_hat, y_true, batch_incident_mask, batch_size):
        # Mask regression prediction for all edges classified as unaffected
        # Done via multiplication with predicted class s.t. it is differentiable
        if self.form == 'incident_only':
            y_true = y_true[batch_incident_mask]
            y_hat = y_hat.reshape(batch_size, -1)
        
            pred_effect_mask = torch.ge(y_hat[...,0], 0) # BCElosswithlogits use sigmoid inside so >= 0 means positive class
            pred_effect_mask = pred_effect_mask.unsqueeze(-1).expand(-1,3)
            y_hat[...,1:] = y_hat[...,1:] * pred_effect_mask

        elif self.form == 'independent':
            y_hat = y_hat.reshape(y_true.shape)
                    
            pred_effect_mask = torch.ge(y_hat[...,0], 0) # BCElosswithlogits use sigmoid inside so >= 0 means positive class
            pred_effect_mask = pred_effect_mask.unsqueeze(-1).expand(-1,-1,3)
            y_hat[...,1:] = y_hat[...,1:] * pred_effect_mask

        elif self.form == 'graph' :
            pred_effect_mask = torch.ge(y_hat[...,0], 0) # BCElosswithlogits use sigmoid inside so >= 0 means positive class
            pred_effect_mask = pred_effect_mask.unsqueeze(-1).expand(-1,-1,3)
            y_hat[...,1:] = y_hat[...,1:] * pred_effect_mask

        return y_hat, y_true

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
