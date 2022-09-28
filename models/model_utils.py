
from doctest import DocTestRunner


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
