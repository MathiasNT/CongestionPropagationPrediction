import torch
import torchmetrics

def generate_masks(test_dataset):
    test_blocked_lanes = test_dataset.incident_info[:,1]

    test_ies = test_dataset.incident_info[:,0]
    one_hot_ies = torch.nn.functional.one_hot(test_ies.to(torch.int64), 147)
    test_ies_n_lanes = test_dataset.network_info[torch.where(one_hot_ies)][:,7:].sum(1)

    test_upstream_mask = test_dataset.network_info[:,:,0] <= 0
    multi_lane_block_mask = (test_blocked_lanes > 1).unsqueeze(1).repeat(1, 147)
    highway_ie_mask = (test_ies_n_lanes >= 3).unsqueeze(1).repeat(1, 147)
    spreading_cong_mask = (test_dataset.target_data[...,0].sum(1) > 1).unsqueeze(1).repeat(1, 147)
    return test_upstream_mask, multi_lane_block_mask, highway_ie_mask, spreading_cong_mask


class MetricObj:
    def __init__(self, bce_pos_weight):
        self.bce_pos_weight = torch.Tensor([bce_pos_weight])


        self.bce_loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=self.bce_pos_weight)
        self.mse_loss_func = torch.nn.MSELoss()
        self.mape_loss_func = torchmetrics.MeanAbsolutePercentageError()
        self.mae_loss_func = torch.nn.L1Loss()
        self.acc_func = torchmetrics.Accuracy()
        self.precision_recall = torchmetrics.functional.precision_recall

    @staticmethod
    def masked_mape(pred, y):
        non_zero_mask = y != 0
        abs_error = (y[non_zero_mask] - pred[non_zero_mask]).abs()
        masked_mape = (abs_error / y[non_zero_mask].abs()).mean()
        return masked_mape 


    def calc_metrics(self, y_hat, y_true, mask=None):

        if mask is None: # Somewhat hacky mask that only reshapes
            mask = torch.ones(y_hat.shape[:2]).bool()

        y_hat = y_hat[mask]        
        y_true = y_true[mask]

        metric_dict = {}
        class_dict  = {}
        start_dict = {}  
        end_dict = {}  
        speed_dict = {}  

        # Classification metrics

        class_dict['bce'] = self.bce_loss_func(y_hat[...,0], y_true[...,0])
        class_dict['acc'] = self.acc_func(y_hat[...,0], y_true[...,0].int())
        precision, recall = self.precision_recall(y_hat[...,0], y_true[...,0].int())
        class_dict['prcsn'] = precision
        class_dict['rcll'] = recall

        # MSE
        start_dict['mse'] = self.mse_loss_func(y_hat[...,1], y_true[...,1])
        end_dict['mse'] = self.mse_loss_func(y_hat[...,2], y_true[...,2])
        speed_dict['mse'] = self.mse_loss_func(y_hat[...,3], y_true[...,3])
    
        # MAE
        start_dict['mae'] = self.mae_loss_func(y_hat[...,1], y_true[...,1])
        end_dict['mae'] = self.mae_loss_func(y_hat[...,2], y_true[...,2])
        speed_dict['mae'] = self.mae_loss_func(y_hat[...,3], y_true[...,3])

        # MAPE
        start_dict['mape'] = self.mape_loss_func(y_hat[...,1], y_true[...,1])
        end_dict['mape'] = self.mape_loss_func(y_hat[...,2], y_true[...,2])
        speed_dict['mape'] = self.mape_loss_func(y_hat[...,3], y_true[...,3])


        # Masked MAPE
        start_dict['Mmape'] = self.masked_mape(y_hat[...,1], y_true[...,1])
        end_dict['Mmape'] = self.masked_mape(y_hat[...,2], y_true[...,2])
        speed_dict['Mmape'] = self.masked_mape(y_hat[...,3], y_true[...,3])


        # Create output dict
        metric_dict['class'] = class_dict
        metric_dict['start'] = start_dict
        metric_dict['end'] = end_dict
        metric_dict['speed'] = speed_dict
        return metric_dict