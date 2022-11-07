import torch
import numpy as np

from doctest import DocTestRunner

from models.rose_models.LGF_model import SimpleGNN, InformedGNN
from models.baselines.lstm import RnnInformedModel, RnnModel, RnnNetworkInformedModel
from models.baselines.mlp import MLPModel
from models.baselines.temporal_cnn import TemporalCNNModel
from models.baselines.lstm_attention import AttentionRNNModel, InformedAttentionRNNModel, NetworkInformedAttentionRNNModel
from models.my_graph.mpnn import MPNN

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_configs(config_dict):

    config = DotDict(config_dict)
    
    # TODO implmement config check here to make it easier for users to create good configs

    return config

def create_gnn_args(config):
    temp_dict = {
        
    'input_dim': config['timeseries_in_size'],
    'output_dim' : config['out_size'],
    'hidden_dim': config['rnn_hidden_size'],
    'num_nodes': config['num_nodes'],
    'input_len': config['n_timesteps'],
    'output_len': 1,

    'max_diffusion_step': config['max_diffusion_step'],
    'filter_type': config['filter_type'],
    'use_gc_ru': config['use_gc_ru'],
    'use_gc_c': config['use_gc_c'],
    'use_curriculum_learning': config['use_curriculum_learning'],
    'cl_decay_steps': config['cl_decay_steps'],
    'dropout': config['dropout'],

    'batch_size': config['batch_size'],
    'num_layers': config['num_layers'],
    #'learning_rate': config.,
    'activation': config['activation'],
    }

    return DotDict(temp_dict)



def init_model(config):
    if config['model'] == 'mlp':
        model = MLPModel(config, learning_rate=config['learning_rate'])

    if config['model'] == 'lstm':
        model = RnnModel(config, learning_rate=config['learning_rate'])

    if config['model'] == 'tcn':
        model = TemporalCNNModel(config, learning_rate=config['learning_rate']) 

    if config['model'] == 'attention':
        model = AttentionRNNModel(config, learning_rate=config['learning_rate']) 

    elif config['model'] == 'informed_lstm':
        model = RnnInformedModel(config, learning_rate=config['learning_rate'])
    
    elif config['model'] == 'network_informed_lstm':
        model = RnnNetworkInformedModel(config, learning_rate=config['learning_rate'])

    elif config['model'] == 'informed_attention':
        model = InformedAttentionRNNModel(config, learning_rate=config['learning_rate'])
    
    elif config['model'] == 'network_informed_attention':
        model = NetworkInformedAttentionRNNModel(config, learning_rate=config['learning_rate'])

    elif config['model'] == 'mpnn_gcn':
        adj_mx =  torch.Tensor(np.load(config['AD_path']))
        model = MLPDecoder(adj_mx=adj_mx, config=config, learning_rate=config['learning_rate'])

    elif config['model'] == 'gnn':
        gnn_args = create_gnn_args(config)
        adj_mx =  torch.Tensor(np.load(config['AD_path']))
        model = SimpleGNN(adj_mx=adj_mx, args=gnn_args, config=config, learning_rate=1e-3) 

    elif config['model'] == 'informed_gnn':
        gnn_args = create_gnn_args(config)
        adj_mx =  torch.Tensor(np.load(config['AD_path']))
        model = InformedGNN(adj_mx=adj_mx, args=gnn_args, config=config, learning_rate=1e-3) 

    return model
