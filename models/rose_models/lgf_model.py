import numpy as np
import torch
from torch import nn
from pytorch_lightning import LightningModule

from ..base_model_class import BaseModelClass
from .lgf_cell import Seq2SeqAttrs, LGFCell


class Encoder(LightningModule, Seq2SeqAttrs):
  """Encoder module.
  Attributes:
    embedding: embedding layer for the input
    lgf_layers: latent graph forecaster layer
  """

  def __init__(self, adj_mx, args):
    super().__init__()

    Seq2SeqAttrs.__init__(self, args)
    self.embedding = nn.Linear(self.input_dim, self.rnn_units)
    torch.nn.init.normal_(self.embedding.weight) # TODO should I comment stuff like this to rose?

    self.lgf_layers = nn.ModuleList(
        [LGFCell(adj_mx, args) for _ in range(self.num_rnn_layers)])
    # self.batch_norm  = nn.BatchNorm1d(self.num_nodes)
    self.dropout = nn.Dropout(self.dropout)
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()

  def forward(self, inputs,
              hidden_state = None):
    """Encoder forward pass.
    Args:
        inputs (tensor): [batch_size, self.num_nodes, self.input_dim]
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
      print('weight nan')
    embedded = self.embedding(inputs)
    embedded = self.tanh(embedded)

    output = self.dropout(embedded)

    if hidden_state is None:
      hidden_state = torch.zeros((self.num_rnn_layers, inputs.shape[0],
                                  self.num_nodes, self.rnn_units),
                                 device=self.device)
    hidden_states = []
    for layer_num, dcgru_layer in enumerate(self.lgf_layers):
      next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
      hidden_states.append(next_hidden_state)
      output = next_hidden_state

    # output = self.batch_norm(output)
    if self.activation == 'relu':
      output = self.relu(output)
    elif self.activation == 'tanh':
      output = self.tanh(output)
    elif self.activation == 'linear':
      pass

    return output, torch.stack(
        hidden_states)  # runs in O(num_layers) so not too slow
        


class SimpleGNN(BaseModelClass, Seq2SeqAttrs):
  """Lightning module for Latent Graph Forecaster model.
  Attributes:
    adj_mx: initialize if learning the graph, load the graph if known
    encoder: encoder module
    decoder: decoder module
  """

  def __init__(self,
               adj_mx,
               args,
               learning_rate,
               config):
    super().__init__(config, learning_rate)
    Seq2SeqAttrs.__init__(self, args)

    print('initialize graph')

    self.register_buffer('adj_mx', adj_mx)

    self.activation = torch.tanh

    self.timeseries_encoder = Encoder(self.adj_mx, args)

    self.fc_shared = nn.Linear(in_features=config['rnn_hidden_size'], out_features=config['fc_hidden_size'])
    self.fc_classifier = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
    self.fc_start = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
    self.fc_end = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
    self.fc_speed = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)

  def forward(self,
              inputs,
               incident_info,
               network_info
              ):
    """LGF forward pass.
    Args:
        inputs: [seq_len, batch_size, num_nodes, input_dim]
        labels: [horizon, batch_size, num_nodes, output_dim]
        batches_seen: batches seen till now
    Returns:
        output: [self.horizon, batch_size, self.num_nodes,
        self.output_dim]
    """

    # reshape [batch, seq_len, num_nodes, dim]
    #           -- > [seq_len, batch, num_nodes, dim]

    inputs = inputs.permute(2, 0, 1, 3)

    encoder_hidden_state = None
    for t in range(self.input_len):
      next_hidden_state, encoder_hidden_state = self.timeseries_encoder(inputs[t], encoder_hidden_state)

    next_hidden_state = self.activation(next_hidden_state)

    hn_fc = self.fc_shared(next_hidden_state)
    hn_fc = self.activation(hn_fc)

    class_logit = self.fc_classifier(hn_fc)

    start_pred = self.fc_start(hn_fc)
    end_pred = self.fc_end(hn_fc)

    speed_pred = self.fc_speed(hn_fc)

    y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
    return y_hat


class InformedGNN(BaseModelClass, Seq2SeqAttrs):
  """Lightning module for Latent Graph Forecaster model.
  Attributes:
    adj_mx: initialize if learning the graph, load the graph if known
    encoder: encoder module
    decoder: decoder module
  """

  def __init__(self,
               adj_mx,
               args,
               learning_rate,
               config):
    super().__init__(config, learning_rate)
    Seq2SeqAttrs.__init__(self, args)

    print('initialize graph')

    self.register_buffer('adj_mx', adj_mx)

    self.activation = torch.tanh

    self.timeseries_encoder = Encoder(self.adj_mx, args)
    self.fc_timeseries = nn.Linear(in_features=config['rnn_hidden_size'], out_features=config['fc_hidden_size'])

    args.use_gc_ru = False # TODO fix this hack
    args.input_dim = config['network_in_size']
    self.incident_encoder = Encoder(self.adj_mx, args)
    self.fc_incident = nn.Linear(in_features=config['rnn_hidden_size'], out_features=config['fc_hidden_size'])
    
    self.fc_shared = nn.Linear(in_features=config['fc_hidden_size'] * 2, out_features=config['fc_hidden_size'])
    self.fc_classifier = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
    self.fc_start = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
    self.fc_end = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)
    self.fc_speed = torch.nn.Linear(in_features=config['fc_hidden_size'], out_features=1)

  def forward(self,
              inputs,
               incident_info,
               network_info
              ):
    """LGF forward pass.
    Args:
        inputs: [seq_len, batch_size, num_nodes, input_dim]
        labels: [horizon, batch_size, num_nodes, output_dim]
        batches_seen: batches seen till now
    Returns:
        output: [self.horizon, batch_size, self.num_nodes,
        self.output_dim]
    """

    # reshape [batch, seq_len, num_nodes, dim]
    #           -- > [seq_len, batch, num_nodes, dim]

    inputs = inputs.permute(2, 0, 1, 3)

    encoder_hidden_state = None
    for t in range(self.input_len):
      next_hidden_state, encoder_hidden_state = self.timeseries_encoder(inputs[t], encoder_hidden_state)
    timeseries_hidden_state = self.activation(next_hidden_state)
    timeseries_hn_fc = self.fc_timeseries(timeseries_hidden_state)
    timeseries_hn_fc = self.activation(timeseries_hn_fc)

    encoder_hidden_state = None
    incident_hidden_state, encoder_hidden_state = self.incident_encoder(network_info, encoder_hidden_state)
    incident_hn_fc = self.fc_incident(incident_hidden_state)
    incident_hn_fc = self.activation(incident_hn_fc)

    hn_fc = self.fc_shared(torch.cat([timeseries_hn_fc, incident_hn_fc], dim=-1))



    class_logit = self.fc_classifier(hn_fc)

    start_pred = self.fc_start(hn_fc)
    end_pred = self.fc_end(hn_fc)

    speed_pred = self.fc_speed(hn_fc)

    y_hat = torch.cat([class_logit, start_pred, end_pred, speed_pred], dim=-1).squeeze()
    return y_hat
