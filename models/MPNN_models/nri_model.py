from models.base_model_class import BaseModelClass
from models.MPNN_models.encoder import MLPEncoder
from models.MPNN_models.decoder import GRUDecoder
from models.MPNN_models.modules import MPNN
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np

from util_folder.ml_utils.loss_utils import KLCategorical


def encode_onehot(labels):
    """This function creates a onehot encoding.
    copied from https://github.com/ethanfetaya/NRI
    """
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


class NRI_v1(BaseModelClass):
    def __init__(self, config, learning_rate, pos_weights):
        super().__init__(config, learning_rate, pos_weights)

        # Generate off-diagonal interaction graph
        self.n_nodes = config["num_nodes"]
        off_diag = np.ones([self.n_nodes, self.n_nodes]) - np.eye(self.n_nodes)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.register_buffer("rel_rec", torch.FloatTensor(rel_rec))
        self.register_buffer("rel_send", torch.FloatTensor(rel_send))

        self.n_edge_types = 1  # TODO check if this should be 1 or 2

        self.burn_in_steps = config["n_timesteps"]

        self.kl_loss = KLCategorical(num_atoms=config["num_nodes"])
        self.log_prior = config["nri_log_prior"]

        self.limited_network_info = config["limited_network_info"]

        self.activation = nn.ReLU()
        encoder_n_in = (
            config["timeseries_in_size"] * config["n_timesteps"] + config["info_in_size"] + config["network_in_size"]
        )
        self.encoder = MLPEncoder(
            n_in=encoder_n_in,
            n_hid=config["nri_n_hid"],
            n_out=self.n_edge_types,
            do_prob=config["dropout"],
            factor=True,
            use_bn=True,
        )
        self.decoder = GRUDecoder(
            n_hid=config["nri_n_hid"],
            f_in=config["timeseries_in_size"],
            msg_hid=config["nri_n_hid"],
            gru_hid=config["nri_n_hid"],
            edge_types=self.n_edge_types,
            skip_first=True,
            do_prob=config["dropout"],
        )
        self.mlp_timeseries = nn.Sequential(
            nn.Linear(in_features=config["rnn_hidden_size"], out_features=config["fc_hidden_size"]),
            self.activation,
            nn.Linear(in_features=config["fc_hidden_size"], out_features=config["rnn_hidden_size"]),
        )
        self.mlp_net_info = nn.Sequential(
            nn.Linear(
                in_features=config["network_in_size"] + config["info_in_size"], out_features=config["fc_hidden_size"]
            ),
            self.activation,
            nn.Linear(in_features=config["fc_hidden_size"], out_features=config["rnn_hidden_size"]),
        )
        self.incident_mpnn1 = MPNN(
            n_in=config["rnn_hidden_size"] * 2,
            n_hid=config["nri_n_hid"],
            n_out=config["nri_n_hid"],
            msg_hid=config["nri_n_hid"],
            msg_out=config["nri_n_hid"],
            n_edge_types=self.n_edge_types,
            dropout_prob=config["dropout"],
        )
        self.incident_mpnn2 = MPNN(
            n_in=config["nri_n_hid"],
            n_hid=config["nri_n_hid"],
            n_out=config["nri_n_hid"],
            msg_hid=config["nri_n_hid"],
            msg_out=config["nri_n_hid"],
            n_edge_types=self.n_edge_types,
            dropout_prob=config["dropout"],
        )
        self.fc_incident = nn.Linear(in_features=config["rnn_hidden_size"], out_features=config["fc_hidden_size"])
        self.mlp_shared = nn.Sequential(
            nn.Linear(in_features=config["fc_hidden_size"], out_features=config["fc_hidden_size"]),
            self.activation,
            nn.Linear(in_features=config["fc_hidden_size"], out_features=config["fc_hidden_size"]),
        )
        self.fc_classifier = torch.nn.Linear(in_features=config["fc_hidden_size"], out_features=1)
        self.fc_start = torch.nn.Linear(in_features=config["fc_hidden_size"], out_features=1)
        self.fc_end = torch.nn.Linear(in_features=config["fc_hidden_size"], out_features=1)
        self.fc_speed = torch.nn.Linear(in_features=config["fc_hidden_size"], out_features=1)

    def forward(self, inputs, incident_info, network_info):

        # reshape [batch, seq_len, num_nodes, dim] -- > [batch, num_nodes, seq_len, dim]
        batch_size, num_nodes, seq_len, input_size = inputs.shape

        encoder_inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        encoder_inci_info = incident_info[..., 1:].unsqueeze(1).repeat(1, num_nodes, 1)
        if self.limited_network_info:
            network_info[..., 0] = network_info[..., 0] == 1
        encoder_full_in = torch.cat([encoder_inputs, encoder_inci_info, network_info], dim=-1)

        edge_logits = self.encoder(encoder_full_in, self.rel_rec, self.rel_send)

        edges = F.gumbel_softmax(
            edge_logits, tau=0.5, hard=True
        )  # TODO fix the gumbel_hard to do the right for different step types
        edge_probs = F.softmax(edge_logits)
        # TODO path out the gumbel_tau

        pred_arr, next_hidden_state = self.decoder(
            inputs,
            self.rel_rec,
            self.rel_send,
            edges,
            burn_in=True,
            burn_in_steps=self.burn_in_steps,
        )

        # Since I don't use preds here as preds I could also use them
        # I could alsouse them as signal for training the GRU
        timeseries_hidden_state = self.activation(next_hidden_state)
        timeseries_hn_fc = self.mlp_timeseries(timeseries_hidden_state)
        timeseries_hn_fc = self.activation(timeseries_hn_fc)

        info = incident_info[:, 1:]  # selecting number of blocked lanes, slow zone speed and duration and startime
        info_mask = torch.nn.functional.one_hot(incident_info[..., 0].to(torch.int64), num_classes=num_nodes).bool()
        padded_info = torch.zeros(batch_size, num_nodes, info.shape[-1], device=self.device)
        padded_info[info_mask] = info

        info_embed = self.mlp_net_info(torch.cat([padded_info, network_info], dim=-1))
        info_embed = torch.relu(info_embed)

        incident_hidden_state = self.incident_mpnn1(
            torch.cat([info_embed, timeseries_hn_fc], dim=-1), self.rel_rec, self.rel_send, edges
        )

        incident_hidden_state = self.incident_mpnn2(incident_hidden_state, self.rel_rec, self.rel_send, edges)

        incident_hn_fc = self.fc_incident(incident_hidden_state)
        incident_hn_fc = self.activation(incident_hn_fc)

        hn_fc = self.mlp_shared(incident_hn_fc)
        class_logit = self.fc_classifier(hn_fc)
        start_pred = self.fc_start(hn_fc)
        end_pred = self.fc_end(hn_fc)
        speed_pred = self.fc_speed(hn_fc)

        y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
        return y_hat, edge_probs

    def standard_step(self, batch, step_type):
        x = batch["input"]
        time = batch["time"]
        y_true = batch["target"]
        incident_info = batch["incident_info"]
        network_info = batch["network_info"].clone()

        batch_size = x.shape[0]
        batch_incident_mask = network_info[:, :, 0] == 0  # Done before transform of network info so allways the same

        x, incident_info, network_info = self.reshape_inputs(x, time, incident_info, network_info, batch_incident_mask)

        # TODO make sure the transform of the network info works for al transforms
        # (IE_only, parrallel and graph)
        if self.transform_network_info_bool:
            network_info = self.transform_network_info(network_info=network_info)

        y_hat, edge_probs = self.forward(inputs=x, incident_info=incident_info, network_info=network_info)

        y_hat, y_true = self.reshape_targets(y_hat, y_true, batch_incident_mask, batch_size)

        bce_loss, start_loss, end_loss, speed_loss = self.calculate_losses(y_hat, y_true)

        assert self.loss_type == "nri_loss", "NRI model should be used w. ELBO-loss (nri_loss)"
        kl_loss = self.kl_loss(preds=edge_probs, log_prior=self.log_prior)
        loss = kl_loss + bce_loss

        if self.full_loss:
            loss += start_loss + end_loss + speed_loss

        self.log(f"{step_type}/kl_divergence", kl_loss, on_step=False, on_epoch=True)

        metrics_dict = self.calc_metrics(y_hat, y_true, step_type)

        self.log(f"{step_type}/loss", loss, on_step=False, on_epoch=True)
        self.log(f"{step_type}/bce_loss", bce_loss, on_step=False, on_epoch=True)
        self.log(f"{step_type}/start_loss", start_loss, on_step=False, on_epoch=True)
        self.log(f"{step_type}/end_loss", end_loss, on_step=False, on_epoch=True)
        self.log(f"{step_type}/speed_loss", speed_loss, on_step=False, on_epoch=True)
        self.log_dict(metrics_dict, on_step=False, on_epoch=True)
        return loss, y_hat.detach(), y_true.detach()


class NRI_uninformed(BaseModelClass):
    def __init__(self, config, learning_rate, pos_weights):
        super().__init__(config, learning_rate, pos_weights)

        # Generate off-diagonal interaction graph
        self.n_nodes = config["num_nodes"]
        off_diag = np.ones([self.n_nodes, self.n_nodes]) - np.eye(self.n_nodes)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.register_buffer("rel_rec", torch.FloatTensor(rel_rec))
        self.register_buffer("rel_send", torch.FloatTensor(rel_send))

        self.n_edge_types = 1  # TODO check if this should be 1 or 2

        self.burn_in_steps = config["n_timesteps"]

        self.kl_loss = KLCategorical(num_atoms=config["num_nodes"])
        self.log_prior = config["nri_log_prior"]

        self.activation = nn.ReLU()
        encoder_n_in = config["timeseries_in_size"] * config["n_timesteps"] + config["network_in_size"]
        self.encoder = MLPEncoder(
            n_in=encoder_n_in,
            n_hid=config["nri_n_hid"],
            n_out=self.n_edge_types,
            do_prob=config["dropout"],
            factor=True,
            use_bn=True,
        )

        self.decoder = GRUDecoder(
            n_hid=config["nri_n_hid"],
            f_in=config["timeseries_in_size"],
            msg_hid=config["nri_n_hid"],
            gru_hid=config["nri_n_hid"],
            edge_types=self.n_edge_types,
            skip_first=True,
            do_prob=config["dropout"],
        )

        self.mlp_timeseries = nn.Sequential(
            nn.Linear(in_features=config["rnn_hidden_size"], out_features=config["fc_hidden_size"]),
            self.activation,
            nn.Linear(in_features=config["fc_hidden_size"], out_features=config["rnn_hidden_size"]),
        )

        self.mlp_net_info = nn.Sequential(
            nn.Linear(in_features=config["network_in_size"], out_features=config["fc_hidden_size"]),
            self.activation,
            nn.Linear(in_features=config["fc_hidden_size"], out_features=config["rnn_hidden_size"]),
        )

        self.incident_mpnn1 = MPNN(
            n_in=config["rnn_hidden_size"] * 2,
            n_hid=config["nri_n_hid"],
            n_out=config["nri_n_hid"],
            msg_hid=config["nri_n_hid"],
            msg_out=config["nri_n_hid"],
            n_edge_types=self.n_edge_types,
            dropout_prob=config["dropout"],
        )

        self.incident_mpnn2 = MPNN(
            n_in=config["nri_n_hid"],
            n_hid=config["nri_n_hid"],
            n_out=config["nri_n_hid"],
            msg_hid=config["nri_n_hid"],
            msg_out=config["nri_n_hid"],
            n_edge_types=self.n_edge_types,
            dropout_prob=config["dropout"],
        )

        self.fc_incident = nn.Linear(in_features=config["rnn_hidden_size"], out_features=config["fc_hidden_size"])

        self.mlp_shared = nn.Sequential(
            nn.Linear(in_features=config["fc_hidden_size"], out_features=config["fc_hidden_size"]),
            self.activation,
            nn.Linear(in_features=config["fc_hidden_size"], out_features=config["fc_hidden_size"]),
        )
        self.fc_classifier = torch.nn.Linear(in_features=config["fc_hidden_size"], out_features=1)
        self.fc_start = torch.nn.Linear(in_features=config["fc_hidden_size"], out_features=1)
        self.fc_end = torch.nn.Linear(in_features=config["fc_hidden_size"], out_features=1)
        self.fc_speed = torch.nn.Linear(in_features=config["fc_hidden_size"], out_features=1)

    def forward(self, inputs, incident_info, network_info):

        # reshape [batch, seq_len, num_nodes, dim] -- > [batch, num_nodes, seq_len, dim]
        batch_size, num_nodes, seq_len, input_size = inputs.shape

        encoder_inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        encoder_inputs  # Create the encoder input here
        network_info_not_incident = network_info[..., 1:]
        encoder_full_in = torch.cat([encoder_inputs, network_info_not_incident], dim=-1)

        edge_logits = self.encoder(encoder_full_in, self.rel_rec, self.rel_send)

        edges = F.gumbel_softmax(
            edge_logits, tau=0.5, hard=True
        )  # TODO fix the gumbel_hard to do the right for different step types
        edge_probs = F.softmax(edge_logits)
        # TODO path out the gumbel_tau

        pred_arr, next_hidden_state = self.decoder(
            inputs,
            self.rel_rec,
            self.rel_send,
            edges,
            burn_in=True,
            burn_in_steps=self.burn_in_steps,
        )

        # Since I don't use preds here as preds I could also use them
        # I could alsouse them as signal for training the GRU
        timeseries_hidden_state = self.activation(next_hidden_state)
        timeseries_hn_fc = self.mlp_timeseries(timeseries_hidden_state)
        timeseries_hn_fc = self.activation(timeseries_hn_fc)

        info_embed = self.mlp_net_info(network_info_not_incident)
        info_embed = torch.relu(info_embed)

        incident_hidden_state = self.incident_mpnn1(
            torch.cat([info_embed, timeseries_hn_fc], dim=-1), self.rel_rec, self.rel_send, edges
        )

        incident_hidden_state = self.incident_mpnn2(incident_hidden_state, self.rel_rec, self.rel_send, edges)

        incident_hn_fc = self.fc_incident(incident_hidden_state)
        incident_hn_fc = self.activation(incident_hn_fc)

        hn_fc = self.mlp_shared(incident_hn_fc)
        class_logit = self.fc_classifier(hn_fc)
        start_pred = self.fc_start(hn_fc)
        end_pred = self.fc_end(hn_fc)
        speed_pred = self.fc_speed(hn_fc)

        y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
        return y_hat, edge_probs

    def standard_step(self, batch, step_type):
        x = batch["input"]
        time = batch["time"]
        y_true = batch["target"]
        incident_info = batch["incident_info"]
        network_info = batch["network_info"].clone()

        batch_size = x.shape[0]
        batch_incident_mask = network_info[:, :, 0] == 0  # Done before transform of network info so allways the same

        x, incident_info, network_info = self.reshape_inputs(x, time, incident_info, network_info, batch_incident_mask)

        # TODO make sure the transform of the network info works for al transforms
        # (IE_only, parrallel and graph)
        if self.transform_network_info_bool:
            network_info = self.transform_network_info(network_info=network_info)

        y_hat, edge_probs = self.forward(inputs=x, incident_info=incident_info, network_info=network_info)

        y_hat, y_true = self.reshape_targets(y_hat, y_true, batch_incident_mask, batch_size)

        bce_loss, start_loss, end_loss, speed_loss = self.calculate_losses(y_hat, y_true)

        assert self.loss_type == "nri_loss", "NRI model should be used w. ELBO-loss (nri_loss)"
        kl_loss = self.kl_loss(preds=edge_probs, log_prior=self.log_prior)
        loss = kl_loss + bce_loss
        self.log(f"{step_type}/kl_divergence", kl_loss, on_step=False, on_epoch=True)

        metrics_dict = self.calc_metrics(y_hat, y_true, step_type)

        self.log(f"{step_type}/loss", loss, on_step=False, on_epoch=True)
        self.log(f"{step_type}/bce_loss", bce_loss, on_step=False, on_epoch=True)
        self.log(f"{step_type}/start_loss", start_loss, on_step=False, on_epoch=True)
        self.log(f"{step_type}/end_loss", end_loss, on_step=False, on_epoch=True)
        self.log(f"{step_type}/speed_loss", speed_loss, on_step=False, on_epoch=True)
        self.log_dict(metrics_dict, on_step=False, on_epoch=True)
        return loss, y_hat.detach(), y_true.detach()


class NRI_uninformed_rw(BaseModelClass):
    def __init__(self, config, learning_rate, pos_weights):
        super().__init__(config, learning_rate, pos_weights)

        # Generate off-diagonal interaction graph
        self.n_nodes = config["num_nodes"]
        off_diag = np.ones([self.n_nodes, self.n_nodes]) - np.eye(self.n_nodes)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.register_buffer("rel_rec", torch.FloatTensor(rel_rec))
        self.register_buffer("rel_send", torch.FloatTensor(rel_send))

        self.n_edge_types = 1  # TODO check if this should be 1 or 2

        self.burn_in_steps = config["n_timesteps"]

        self.kl_loss = KLCategorical(num_atoms=config["num_nodes"])
        self.log_prior = config["nri_log_prior"]

        self.activation = nn.ReLU()
        encoder_n_in = config["timeseries_in_size"] * config["n_timesteps"]
        self.encoder = MLPEncoder(
            n_in=encoder_n_in,
            n_hid=config["nri_n_hid"],
            n_out=self.n_edge_types,
            do_prob=config["dropout"],
            factor=True,
            use_bn=True,
        )

        self.decoder = GRUDecoder(
            n_hid=config["nri_n_hid"],
            f_in=config["timeseries_in_size"],
            msg_hid=config["nri_n_hid"],
            gru_hid=config["nri_n_hid"],
            edge_types=self.n_edge_types,
            skip_first=True,
            do_prob=config["dropout"],
        )

        self.mlp_timeseries = nn.Sequential(
            nn.Linear(in_features=config["rnn_hidden_size"], out_features=config["fc_hidden_size"]),
            self.activation,
            nn.Linear(in_features=config["fc_hidden_size"], out_features=config["rnn_hidden_size"]),
        )

        self.incident_mpnn1 = MPNN(
            n_in=config["rnn_hidden_size"],
            n_hid=config["nri_n_hid"],
            n_out=config["nri_n_hid"],
            msg_hid=config["nri_n_hid"],
            msg_out=config["nri_n_hid"],
            n_edge_types=self.n_edge_types,
            dropout_prob=config["dropout"],
        )

        self.incident_mpnn2 = MPNN(
            n_in=config["nri_n_hid"],
            n_hid=config["nri_n_hid"],
            n_out=config["nri_n_hid"],
            msg_hid=config["nri_n_hid"],
            msg_out=config["nri_n_hid"],
            n_edge_types=self.n_edge_types,
            dropout_prob=config["dropout"],
        )

        self.fc_incident = nn.Linear(in_features=config["rnn_hidden_size"], out_features=config["fc_hidden_size"])

        self.mlp_shared = nn.Sequential(
            nn.Linear(in_features=config["fc_hidden_size"], out_features=config["fc_hidden_size"]),
            self.activation,
            nn.Linear(in_features=config["fc_hidden_size"], out_features=config["fc_hidden_size"]),
        )
        self.fc_classifier = torch.nn.Linear(in_features=config["fc_hidden_size"], out_features=1)
        self.fc_start = torch.nn.Linear(in_features=config["fc_hidden_size"], out_features=1)
        self.fc_end = torch.nn.Linear(in_features=config["fc_hidden_size"], out_features=1)
        self.fc_speed = torch.nn.Linear(in_features=config["fc_hidden_size"], out_features=1)

    def forward(self, inputs, incident_info, network_info):

        # reshape [batch, seq_len, num_nodes, dim] -- > [batch, num_nodes, seq_len, dim]
        batch_size, num_nodes, seq_len, input_size = inputs.shape

        encoder_inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        encoder_inputs  # Create the encoder input here
        network_info_not_incident = network_info[..., 1:]
        encoder_full_in = torch.cat([encoder_inputs, network_info_not_incident], dim=-1)

        edge_logits = self.encoder(encoder_full_in, self.rel_rec, self.rel_send)

        edges = F.gumbel_softmax(
            edge_logits, tau=0.5, hard=True
        )  # TODO fix the gumbel_hard to do the right for different step types
        edge_probs = F.softmax(edge_logits)
        # TODO path out the gumbel_tau

        pred_arr, next_hidden_state = self.decoder(
            inputs,
            self.rel_rec,
            self.rel_send,
            edges,
            burn_in=True,
            burn_in_steps=self.burn_in_steps,
        )

        # Since I don't use preds here as preds I could also use them
        # I could alsouse them as signal for training the GRU
        timeseries_hidden_state = self.activation(next_hidden_state)
        timeseries_hn_fc = self.mlp_timeseries(timeseries_hidden_state)
        timeseries_hn_fc = self.activation(timeseries_hn_fc)

        incident_hidden_state = self.incident_mpnn1(timeseries_hn_fc, self.rel_rec, self.rel_send, edges)

        incident_hidden_state = self.incident_mpnn2(incident_hidden_state, self.rel_rec, self.rel_send, edges)

        incident_hn_fc = self.fc_incident(incident_hidden_state)
        incident_hn_fc = self.activation(incident_hn_fc)

        hn_fc = self.mlp_shared(incident_hn_fc)
        class_logit = self.fc_classifier(hn_fc)
        start_pred = self.fc_start(hn_fc)
        end_pred = self.fc_end(hn_fc)
        speed_pred = self.fc_speed(hn_fc)

        y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
        return y_hat, edge_probs

    def standard_step(self, batch, step_type):
        x = batch["input"]
        time = batch["time"]
        y_true = batch["target"]
        incident_info = batch["incident_info"]
        network_info = batch["network_info"].clone()

        batch_size = x.shape[0]
        batch_incident_mask = network_info[:, :, 0] == 0  # Done before transform of network info so allways the same

        x, incident_info, network_info = self.reshape_inputs(x, time, incident_info, network_info, batch_incident_mask)

        # TODO make sure the transform of the network info works for al transforms
        # (IE_only, parrallel and graph)
        if self.transform_network_info_bool:
            network_info = self.transform_network_info(network_info=network_info)

        y_hat, edge_probs = self.forward(inputs=x, incident_info=incident_info, network_info=network_info)

        y_hat, y_true = self.reshape_targets(y_hat, y_true, batch_incident_mask, batch_size)

        bce_loss, start_loss, end_loss, speed_loss = self.calculate_losses(y_hat, y_true)

        assert self.loss_type == "nri_loss", "NRI model should be used w. ELBO-loss (nri_loss)"
        kl_loss = self.kl_loss(preds=edge_probs, log_prior=self.log_prior)
        loss = kl_loss + bce_loss
        self.log(f"{step_type}/kl_divergence", kl_loss, on_step=False, on_epoch=True)

        metrics_dict = self.calc_metrics(y_hat, y_true, step_type)

        self.log(f"{step_type}/loss", loss, on_step=False, on_epoch=True)
        self.log(f"{step_type}/bce_loss", bce_loss, on_step=False, on_epoch=True)
        self.log(f"{step_type}/start_loss", start_loss, on_step=False, on_epoch=True)
        self.log(f"{step_type}/end_loss", end_loss, on_step=False, on_epoch=True)
        self.log(f"{step_type}/speed_loss", speed_loss, on_step=False, on_epoch=True)
        self.log_dict(metrics_dict, on_step=False, on_epoch=True)
        return loss, y_hat.detach(), y_true.detach()
