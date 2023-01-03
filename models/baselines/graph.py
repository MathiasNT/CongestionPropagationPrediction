import pytorch_lightning as pl
import torch
import torchmetrics
from torchmetrics.functional import precision_recall


class GraphBase_unused(pl.LightningModule):
    def __init__(self, config, learning_rate, bce_pos_weight):
        super().__init__()
        self.learning_rate = learning_rate
        self.incident_only = config["incident_only"]
        self.bce_pos_weight = torch.Tensor([bce_pos_weight])
        self.bce_loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=self.bce_pos_weight)
        self.mse_loss_func = torch.nn.MSELoss()
        self.acc_func = torchmetrics.Accuracy()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x = batch["input"]
        time = batch["time"]
        y_true = batch["target"]
        incident_info = batch["incident_info"]
        network_info = batch["network_info"]

        batch_size = x.shape[0]
        batch_incident_mask = network_info[:, :, 0] == 0

        x, incident_info, network_info = self.reshape_inputs(x, time, incident_info, network_info, batch_incident_mask)

        y_hat = self.forward(x, incident_info, network_info)

        y_hat, y_true = self.reshape_targets(y_hat, y_true, batch_incident_mask, batch_size)

        bce_loss, start_loss, end_loss, speed_loss = self.get_combined_loss(y_hat, y_true)
        loss = bce_loss + start_loss + end_loss + speed_loss
        accuracy = self.acc_func(y_hat[..., 0], y_true[..., 0].int())
        precision, recall = precision_recall(y_hat[..., 0], y_true[..., 0].int())

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/bce_loss", bce_loss, on_step=False, on_epoch=True)
        self.log("train/start_loss", start_loss, on_step=False, on_epoch=True)
        self.log("train/end_loss", end_loss, on_step=False, on_epoch=True)
        self.log("train/speed_loss", speed_loss, on_step=False, on_epoch=True)
        self.log("train/accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("train/precision", precision, on_step=False, on_epoch=True)
        self.log("train/recall", recall, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["input"]
        time = batch["time"]
        y_true = batch["target"]
        incident_info = batch["incident_info"]
        network_info = batch["network_info"]

        batch_size = x.shape[0]
        batch_incident_mask = network_info[:, :, 0] == 0

        x, incident_info, network_info = self.reshape_inputs(x, time, incident_info, network_info, batch_incident_mask)

        y_hat = self.forward(x, incident_info, network_info)

        y_hat, y_true = self.reshape_targets(y_hat, y_true, batch_incident_mask, batch_size)

        bce_loss, start_loss, end_loss, speed_loss = self.get_combined_loss(y_hat, y_true)
        loss = bce_loss + start_loss + end_loss + speed_loss
        accuracy = self.acc_func(y_hat[..., 0], y_true[..., 0].int())
        precision, recall = precision_recall(y_hat[..., 0], y_true[..., 0].int())

        self.log("val/loss", loss, on_epoch=True)
        self.log("val/bce_loss", bce_loss, on_epoch=True)
        self.log("val/start_loss", start_loss, on_epoch=True)
        self.log("val/end_loss", end_loss, on_epoch=True)
        self.log("val/speed_loss", speed_loss, on_epoch=True)
        self.log("val/accuracy", accuracy, on_epoch=True)
        self.log("val/precision", precision, on_epoch=True)
        self.log("val/recall", recall, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # TODO double check if this function is necessary, I think PL will just use the val function
        x = batch["input"]
        time = batch["time"]
        y_true = batch["target"]
        incident_info = batch["incident_info"]
        network_info = batch["network_info"]

        batch_size = x.shape[0]
        batch_incident_mask = network_info[:, :, 0] == 0

        x, incident_info, network_info = self.reshape_inputs(x, time, incident_info, network_info, batch_incident_mask)

        y_hat = self.forward(x, incident_info, network_info)

        y_hat, y_true = self.reshape_targets(y_hat, y_true, batch_incident_mask, batch_size)

        bce_loss, start_loss, end_loss, speed_loss = self.get_combined_loss(y_hat, y_true)
        loss = bce_loss + start_loss + end_loss + speed_loss
        accuracy = self.acc_func(y_hat[..., 0], y_true[..., 0].int())
        precision, recall = precision_recall(y_hat[..., 0], y_true[..., 0].int())

        self.log("test/loss", loss, on_epoch=True)
        self.log("test/bce_loss", bce_loss, on_epoch=True)
        self.log("test/start_loss", start_loss, on_epoch=True)
        self.log("test/end_loss", end_loss, on_epoch=True)
        self.log("test/speed_loss", speed_loss, on_epoch=True)
        self.log("test/accuracy", accuracy, on_epoch=True)
        self.log("test/precision", precision, on_epoch=True)
        self.log("test/recall", recall, on_epoch=True)

    def standard_step(self, batch, step_type):
        # TODO Tried this but the logging got wonky - don't know why though which is weird
        x = batch["input"]
        time = batch["time"]
        y_true = batch["target"]
        incident_info = batch["incident_info"]
        network_info = batch["network_info"]

        batch_size = x.shape[0]
        batch_incident_mask = network_info[:, :, 0] == 0

        x, incident_info, network_info = self.reshape_inputs(x, time, incident_info, network_info, batch_incident_mask)

        y_hat = self.forward(x, incident_info, network_info)

        y_hat, y_true = self.reshape_targets(y_hat, y_true, batch_incident_mask, batch_size)

        bce_loss, start_loss, end_loss, speed_loss = self.get_combined_loss(y_hat, y_true)
        loss = bce_loss + start_loss + end_loss + speed_loss
        accuracy = self.acc_func(y_hat[..., 0], y_true[..., 0].int())
        precision, recall = precision_recall(y_hat[..., 0], y_true[..., 0].int())

        self.log(f"{step_type}/loss", loss, on_epoch=True)
        self.log(f"{step_type}/bce_loss", bce_loss, on_epoch=True)
        self.log(f"{step_type}/start_loss", start_loss, on_epoch=True)
        self.log(f"{step_type}/end_loss", end_loss, on_epoch=True)
        self.log(f"{step_type}/speed_loss", speed_loss, on_epoch=True)
        self.log(f"{step_type}/accuracy", accuracy, on_epoch=True)
        self.log(f"{step_type}/precision", precision, on_epoch=True)
        self.log(f"{step_type}/recall", recall, on_epoch=True)
        return loss

    def get_combined_loss(self, y_hat, y):
        assert y_hat.shape == y.shape, "Shapes do not match"
        bce_loss = self.bce_loss_func(y_hat[..., 0], y[..., 0])
        start_loss = self.mse_loss_func(y_hat[..., 1], y[..., 1])
        end_loss = self.mse_loss_func(y_hat[..., 2], y[..., 2])
        speed_loss = self.mse_loss_func(y_hat[..., 3], y[..., 3])
        return bce_loss, start_loss, end_loss, speed_loss

    def reshape_inputs(self, x, time, incident_info, network_info, batch_incident_mask):
        batch_size, n_nodes, n_lanes, n_timesteps, n_obs_features = x.shape
        n_time_features = time.shape[-1]

        if self.incident_only:
            x = x[batch_incident_mask]
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(batch_size, n_timesteps, n_lanes * n_obs_features)
            time = time[:, 0, 0, :, :]
            x = torch.cat([x, time], dim=-1)
        else:
            x = x.permute(0, 1, 3, 2, 4)
            x = x.reshape(batch_size, n_nodes, n_timesteps, n_lanes * n_obs_features)
            time = time[:, :, 0, :, :]
            x = torch.cat([x, time], dim=-1)
            x = x.reshape(-1, n_timesteps, n_lanes * n_obs_features + n_time_features)
            incident_info = incident_info.unsqueeze(1).expand(-1, n_nodes, -1)
            incident_info = incident_info.reshape(-1, incident_info.shape[-1])
            network_info = network_info[..., 0].reshape(-1)

        return x, incident_info, network_info

    def reshape_targets(self, y_hat, y_true, batch_incident_mask, batch_size):
        if self.incident_only:
            y_true = y_true[batch_incident_mask]
            y_hat = y_hat.reshape(batch_size, -1)
        else:
            y_hat = y_hat.reshape(y_true.shape)
        return y_hat, y_true

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
