# TODO Find a better name for this file

import pytorch_lightning as pl
import torch
from torch import nn

from .lgf_cell import Seq2SeqAttrs, LGFCell
from ..base_model_class import BaseModelClass


class LFGModel(BaseModelClass, Seq2SeqAttrs):
    """Rewritten Encoder module. TODO clean up
    Attributes:
      embedding: embedding layer for the input
      lgf_layers: latent graph forecaster layer
    """

    def __init__(self, adj_mx, args, config, learning_rate=1e-3):
        super().__init__(config, learning_rate)

        Seq2SeqAttrs.__init__(self, args)
        self.embedding = nn.Linear(self.input_dim, self.rnn_units)
        torch.nn.init.normal_(self.embedding.weight)

        self.lgf_layers = nn.ModuleList([LGFCell(adj_mx, args) for _ in range(self.num_rnn_layers)])
        # self.batch_norm  = nn.BatchNorm1d(self.num_nodes)
        self.dropout = nn.Dropout(self.dropout)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, inputs, hidden_state=None):
        """Encoder forward pass.
        Args:
            inputs ( tensor): [batch_size, self.num_nodes, self.input_dim]
            hidden_state (tensor): [num_layers, batch_size, self.rnn_units]
              optional, zeros if not provided
        Returns:
            output: # shape (batch_size, num_nodes,  self.rnn_units)
            hidden_state # shape (num_layers, batch_size, num_nodes,
            self.rnn_units)
              (lower indices mean lower layers)
        """
        linear_weights = self.embedding.weight
        if torch.any(torch.isnan(linear_weights)):
            print("weight nan")
        embedded = self.embedding(inputs)
        embedded = self.tanh(embedded)

        output = self.dropout(embedded)

        if hidden_state is None:
            hidden_state = torch.zeros(
                (self.num_rnn_layers, self.batch_size, self.num_nodes, self.rnn_units), device=self.device
            )
        hidden_states = []
        for layer_num, dcgru_layer in enumerate(self.lgf_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        # output = self.batch_norm(output)
        if self.activation == "relu":
            output = self.relu(output)
        elif self.activation == "tanh":
            output = self.tanh(output)
        elif self.activation == "linear":
            pass

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow
