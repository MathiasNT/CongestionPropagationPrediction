
from models.base_model_class import BaseModelClass
from models.MPNN_models.encoder import FixedEncoder
from models.MPNN_models.decoder import GRUDecoder
from models.MPNN_models.modules import MPNN 
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np


def encode_onehot(labels):
    """This function creates a onehot encoding.
    copied from https://github.com/ethanfetaya/NRI
    """
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

class InformedMPNNModel(BaseModelClass):
    def __init__(self, adj_mx, config, learning_rate, pos_weights):
        super().__init__(config, learning_rate, pos_weights)
        
        # Generate off-diagonal interaction graph
        self.n_nodes = config['num_nodes']
        off_diag = np.ones([self.n_nodes, self.n_nodes]) - np.eye(self.n_nodes)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.register_buffer('rel_rec', torch.FloatTensor(rel_rec))
        self.register_buffer('rel_send', torch.FloatTensor(rel_send))

        self.burn_in_steps = config['n_timesteps']

        self.activation = nn.ReLU()

        self.adj_to_rel_types = FixedEncoder(adj_matrix=adj_mx)
        self.n_edge_types = 1 #TODO check if this should be 1 or 2 

        self.decoder = GRUDecoder(
                n_hid=config['nri_n_hid'],
                f_in=config['timeseries_in_size'],
                msg_hid=config['nri_n_hid'],
                gru_hid=config['nri_n_hid'],
                edge_types=self.n_edge_types, 
                skip_first=True,
                do_prob=config['dropout'],
            )

        self.mlp_timeseries = nn.Sequential(
                                            nn.Linear(in_features=config['rnn_hidden_size'],
                                                      out_features=config['fc_hidden_size']),
                                            self.activation,
                                            nn.Linear(in_features=config['fc_hidden_size'],
                                                      out_features=config['rnn_hidden_size']),
                                            )

        self.mlp_net_info = nn.Sequential( 
                                          nn.Linear(in_features=config['network_in_size'] + config['info_in_size'],
                                                    out_features=config['fc_hidden_size']),
                                          self.activation,
                                          nn.Linear(in_features=config['fc_hidden_size'],
                                                    out_features=config['rnn_hidden_size']),
                                         )
        

        self.incident_mpnn1 = MPNN(n_in=config['rnn_hidden_size'] * 2,
                                  n_hid=config['nri_n_hid'],
                                  n_out=config['nri_n_hid'],
                                  msg_hid=config['nri_n_hid'],
                                  msg_out=config['nri_n_hid'],
                                  n_edge_types=self.n_edge_types,
                                  dropout_prob=config['dropout']) 

        self.incident_mpnn2 = MPNN(n_in=config['nri_n_hid'],
                                  n_hid=config['nri_n_hid'],
                                  n_out=config['nri_n_hid'],
                                  msg_hid=config['nri_n_hid'],
                                  msg_out=config['nri_n_hid'],
                                  n_edge_types=self.n_edge_types,
                                  dropout_prob=config['dropout']) 

        self.fc_incident = nn.Linear(in_features=config['rnn_hidden_size'], out_features=config['fc_hidden_size'])

        self.mlp_shared = nn.Sequential(
                                        nn.Linear(in_features=config['fc_hidden_size'],
                                                  out_features=config['fc_hidden_size']),
                                        self.activation,
                                        nn.Linear(in_features=config['fc_hidden_size'],
                                                  out_features=config['fc_hidden_size'])
                                        )
        self.fc_classifier = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_start = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_end = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
        self.fc_speed = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)

    def forward(self, inputs, incident_info, network_info):

        # reshape [batch, seq_len, num_nodes, dim] -- > [batch, num_nodes, seq_len, dim]
        batch_size, num_nodes, seq_len, input_size = inputs.shape
        # This here is just a quickfix an updated function would be bette
        edge_logits = self.adj_to_rel_types(inputs, self.rel_rec, self.rel_send)

        edges = (edge_logits == 0).long() 

        pred_arr, next_hidden_state = self.decoder(inputs,
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
        
        info = incident_info[:,1:]   # selecting number of blocked lanes, slow zone speed and duration and startime 
        info_mask = torch.nn.functional.one_hot(incident_info[...,0].to(torch.int64), num_classes=num_nodes).bool()
        padded_info = torch.zeros(batch_size, num_nodes, info.shape[-1], device=self.device)
        padded_info[info_mask] = info

        info_embed = self.mlp_net_info(torch.cat([padded_info, network_info], dim=-1))
        info_embed = torch.relu(info_embed)

        incident_hidden_state = self.incident_mpnn1(torch.cat([info_embed, timeseries_hn_fc], dim=-1),
                                                   self.rel_rec,
                                                   self.rel_send,
                                                   edges)
 
        incident_hidden_state = self.incident_mpnn2(incident_hidden_state,
                                                   self.rel_rec,
                                                   self.rel_send,
                                                   edges)

        incident_hn_fc = self.fc_incident(incident_hidden_state)
        incident_hn_fc = self.activation(incident_hn_fc)

        hn_fc = self.mlp_shared(incident_hn_fc)
        class_logit = self.fc_classifier(hn_fc)
        start_pred = self.fc_start(hn_fc)
        end_pred = self.fc_end(hn_fc)
        speed_pred = self.fc_speed(hn_fc)

        y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
        return y_hat