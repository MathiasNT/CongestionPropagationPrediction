from models.base_model_class import BaseModelClass
from models.MPNN_models.encoder import MLPEncoder
from models.MPNN_models.decoder import GRUDecoder
import torch.nn.functional as F
import torch
from torch import nn


class NRI_model(BaseModelClass):

    def __init__(self, config, learning_rate, pos_weights):
        super().__init__(self, config, learning_rate, pos_weights)
        self.gubel_tau = 
        self.gumbel_hard = 
        self.rel_rec = 
        self.rel_send = 
        self.burn_in = 

        self.activation = 

        self.encoder = MLPEncoder(n_in=, 
                                  n_hid=, 
                                  n_out=, 
                                  do_prob=config['dropout_p'], 
                                  factor=config['nri_factor'], 
                                  use_bn=config['nri_bn'],)

        self.decoder = GRUDecoder(
                n_hid=,
                f_in=,
                msg_hid=,
                gru_hid=,
                edge_types=config['nri_edge_types'],
                skip_first=config['nri_skip_first'],
                do_prob=config['dropout_p'],
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
        batch_size, seq_len, num_nodes, input_size = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3)



        edge_logits = self.encoder(inputs, self.rel_rec, self.rel_send)

        edges = F.gumbel_softmax(edge_logits, tau=self.gumbel_tau, hard=self.gumbel_hard)

        pred_arr = self.decoder(inputs,
                                self.rel_rec,
                                self.rel_send,
                                edges,
                                burn_in=self.burn_in,
                                burn_in_steps=self.burn_in_steps,
                                split_len=self.split_len,
                                )

        y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
        return y_hat