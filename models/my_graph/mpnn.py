# TODO This implementation use so much memory. Come back to this if Rose implementation also does that
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from ..base_model_class import BaseModelClass


def edge2node(x, rel_rec):
    """This function makes the aggregation over the incomming edge embeddings"""
    incoming = torch.matmul(rel_rec.t(), x)
    return incoming / incoming.size(1)


def node2edge(x, rel_rec, rel_send):
    """This function makes a matrix of [node_i, node_j] rows for the edge embeddings"""
    receivers = torch.matmul(rel_rec, x)
    senders = torch.matmul(rel_send, x)
    edges = torch.cat([senders, receivers], dim=-1)
    return edges


def encode_onehot(labels):
    """This function creates a onehot encoding.
    copied from https://github.com/ethanfetaya/NRI
    """
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def adj_to_rel_type(rel_rec, rel_send, adj):
    """Go from a (sender, receiver) ordered adj matrix to rel_type
        for a MPNN model    

    Args:
        rel_rec (torch.Tensor [N(N-1), N]): tensor w. one-hot-encoded idx of receiver pr edge
        rel_send (torch.Tensor [N(N-1), N]): tensor w. one-hot-encoded idx of receiver pr edge
        adj (torch.Tensor [N, N]): (sender, receiver) ordered adj matrix) 

    Returns:
        torch.Tensor [N(N-1)]: or of rel types 
    """

    rel_type = torch.zeros(rel_rec.shape[0])
    for i, (rec, send) in enumerate(zip(rel_rec, rel_send)):
        rec_idx = torch.where(rec)[0]
        sender_idx = torch.where(send)[0]
        rel_type[i] = adj[sender_idx, rec_idx]
    return rel_type


class MLPDecoder(BaseModelClass):
    """Simple non recurrent decoder, based on NRI paper."""

    def __init__(self, adj_mx, config, learning_rate=1e-3):
        super().__init__(config, learning_rate)

        self.edge_types = 2 # This will only be used if we do a full NRI model
        self.dropout_prob = config['dropout']

        n_in = config['timeseries_in_size']
        msg_hid = config['rnn_hidden_size']
        msg_out = config['rnn_hidden_size']
        n_hid = config['fc_hidden_size']
        n_out = config['out_size']
        n_timesteps = config['n_timesteps']


        # FC layers to compute messages
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(in_features=n_in * 2 * n_timesteps, out_features=msg_hid) for _ in range(self.edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(in_features=msg_hid, out_features=msg_out) for _ in range(self.edge_types)]
        )

        # FC for generating the output
        self.out_fc1 = nn.Linear(in_features=msg_out + n_in * n_timesteps, out_features=n_hid)
        self.out_fc2 = nn.Linear(in_features=n_hid, out_features=n_hid)
        self.out_fc3 = nn.Linear(in_features=n_hid, out_features=n_out)

        self.msg_out_shape = msg_out

        
        n_nodes = adj_mx.shape[0]
        off_diag = np.ones([n_nodes, n_nodes]) - np.eye(n_nodes)
        rel_rec = torch.Tensor(encode_onehot(np.where(off_diag)[0]))
        rel_send = torch.Tensor(encode_onehot(np.where(off_diag)[1])) # TODO double check how the order of the ADJ should be
        self.register_buffer('rel_rec', rel_rec)
        self.register_buffer('rel_send', rel_send)

        rel_types = adj_to_rel_type(rel_rec=rel_rec, rel_send=rel_send, adj=adj_mx)
        self.register_buffer('rel_types', rel_types)


    def forward(self, x, incident_info, network_info):
        # input shape [batch_size, num_timesteps, num_atoms, num_dims]
        # rel_types [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]
        # Since we do a simplified version timesteps here are considered dims so permute to get [B, N, T/F]
        batch_size = x.shape[0]
        n_nodes = x.shape[1]
        n_timesteps = x.shape[2]
        n_features = x.shape[3]

        x = x.reshape(batch_size, n_nodes, -1)

        pre_msg = node2edge(x, self.rel_rec, self.rel_send)

        # Create variable to aggregate the messages in
        all_msgs = Variable(
            torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape, device=x.device) # TODO any way to do the device here in a more PL way?
        )

        # Go over the different edge types and compute their contribution to the overall messages
        rel_types = self.rel_types.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)        


        for i in range(0, self.edge_types):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * rel_types
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_msgs = torch.cat([x, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(
            F.relu(self.out_fc1(aug_msgs)), p=self.dropout_prob, training=self.training
        )
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob, training=self.training)
        pred = self.out_fc3(pred)

        return pred


class GRUDecoder(nn.Module):
    """summary"""

    def __init__(self, n_hid, f_in, msg_hid, gru_hid, edge_types, skip_first, do_prob):
        super().__init__()

        self.edge_types = edge_types

        # FC layers to compute messages
        self.msg_fc1 = nn.ModuleList(
            [
                nn.Linear(in_features=gru_hid * 2, out_features=msg_hid)
                for _ in range(self.edge_types)  # 2*n_hid, n_hid is their implementation
            ]
        )
        self.msg_fc2 = nn.ModuleList(
            [
                nn.Linear(in_features=msg_hid, out_features=msg_hid)
                for _ in range(self.edge_types)  # n_hid, n_hid is their implementation
            ]
        )

        self.msg_out_shape = msg_hid  # They have n_hid here

        # GRU network
        self.gru_hr = nn.Linear(in_features=msg_hid, out_features=gru_hid, bias=False)
        self.gru_hi = nn.Linear(in_features=msg_hid, out_features=gru_hid, bias=False)
        self.gru_hn = nn.Linear(in_features=msg_hid, out_features=gru_hid, bias=False)

        self.gru_ir = nn.Linear(in_features=f_in, out_features=gru_hid)
        self.gru_ii = nn.Linear(in_features=f_in, out_features=gru_hid)
        self.gru_in = nn.Linear(in_features=f_in, out_features=gru_hid)

        # FC for generating the output
        self.out_fc1 = nn.Linear(in_features=gru_hid, out_features=n_hid)
        self.out_fc2 = nn.Linear(in_features=n_hid, out_features=n_hid)
        self.out_fc3 = nn.Linear(in_features=n_hid, out_features=f_in)
        self.gru_hid = gru_hid

        self.skip_first = skip_first
        self.dropout_prob = do_prob

    def do_single_step_forward(self, inputs, rel_rec, rel_send, rel_types, hidden):

        # input shape [batch_size, num_timesteps, num_atoms, num_dims]
        # rel_types [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]
        pre_msg = node2edge(hidden, rel_rec, rel_send)

        # Create variable to aggregate the messages in
        all_msgs = Variable(
            torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape, device=inputs.device)
        )

        if self.skip_first:
            start_idx = 1
        else:
            start_idx = 0
        # Go over the different edge types and compute their contribution to the overall messages
        for i in range(start_idx, self.edge_types):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob, training=self.training)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_types[:, :, i : i + 1]
            all_msgs += msg / float(self.edge_types)

        # mean all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs / agg_msgs.shape[1]

        # Send through GRU network
        r = torch.sigmoid(self.gru_ir(inputs) + self.gru_hr(agg_msgs))
        i = torch.sigmoid(self.gru_ii(inputs) + self.gru_hi(agg_msgs))
        n = torch.tanh(self.gru_in(inputs) + r * self.gru_hn(agg_msgs))

        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob, training=self.training)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob, training=self.training)
        pred = self.out_fc3(pred)

        # Do a skip connection
        assert inputs.shape == pred.shape, "Input feature dim should match output feature dim"
        pred = inputs + pred

        return pred, hidden

    def forward(
        self,
        inputs,
        rel_rec,
        rel_send,
        rel_types,
        burn_in,
        burn_in_steps,
        split_len,
    ):
        # Inputs should be [B, T, N, F]

        pred_all = []

        hidden = Variable(
            torch.zeros(inputs.size(0), inputs.size(2), self.gru_hid, device=inputs.device)
        )

        for step in range(0, inputs.shape[1] - 1):
            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            pred, hidden = self.do_single_step_forward(ins, rel_rec, rel_send, rel_types, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds


class GRUDecoder_global(BaseModelClass):
    """summary"""

    def __init__(self, n_hid, f_in, msg_hid, gru_hid, edge_types, skip_first, do_prob, w_in=2):
        super().__init__()

        self.edge_types = edge_types

        # FC layers to compute messages
        self.msg_fc1 = nn.ModuleList(
            [
                nn.Linear(in_features=gru_hid * 2 + w_in, out_features=msg_hid)
                for _ in range(self.edge_types)  # 2*n_hid, n_hid is their implementation
            ]
        )
        self.msg_fc2 = nn.ModuleList(
            [
                nn.Linear(in_features=msg_hid, out_features=msg_hid)
                for _ in range(self.edge_types)  # n_hid, n_hid is their implementation
            ]
        )

        self.msg_out_shape = msg_hid  # They have n_hid here

        # GRU network
        # They have n_hid, n_hid for all of these
        self.gru_hr = nn.Linear(in_features=msg_hid, out_features=gru_hid, bias=False)
        self.gru_hi = nn.Linear(in_features=msg_hid, out_features=gru_hid, bias=False)
        self.gru_hn = nn.Linear(in_features=msg_hid, out_features=gru_hid, bias=False)

        # They have n_in_node, n_hid for all of these
        self.gru_ir = nn.Linear(in_features=f_in + w_in, out_features=gru_hid)
        self.gru_ii = nn.Linear(in_features=f_in + w_in, out_features=gru_hid)
        self.gru_in = nn.Linear(in_features=f_in + w_in, out_features=gru_hid)

        # FC for generating the output
        # n_hid, n_hid
        self.out_fc1 = nn.Linear(in_features=gru_hid, out_features=n_hid)
        self.out_fc2 = nn.Linear(in_features=n_hid, out_features=n_hid)  # n_hid, n_hid
        # n_hid,n_in_node
        self.out_fc3 = nn.Linear(in_features=n_hid, out_features=f_in)
        self.gru_hid = gru_hid

        self.skip_first = skip_first
        self.dropout_prob = do_prob

    def do_single_step_forward(self, inputs, global_ins, rel_rec, rel_send, rel_types, hidden):

        # input shape [batch_size, num_timesteps, num_atoms, num_dims]
        # rel_types [batch_size, num_atoms*(num_atoms-1), num_edge_types]
        pre_msg = node2edge(hidden, rel_rec, rel_send)

        # Add the global information to the premessage
        global_ins = global_ins.unsqueeze(1)
        pre_msg = torch.cat([pre_msg, global_ins.repeat(1, pre_msg.size(1), 1)], dim=-1)

        # Create variable to aggregate the messages in
        all_msgs = Variable(
            torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape, device=inputs.device)
        )

        if self.skip_first:
            start_idx = 1
        else:
            start_idx = 0
        # Go over the different edge types and compute their contribution to the overall messages
        for i in range(start_idx, self.edge_types):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob, training=self.training)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_types[:, :, i : i + 1]
            all_msgs += msg / float(self.edge_types)

        # mean all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs / agg_msgs.shape[1]

        # Send through GRU network
        combined_inputs = torch.cat([inputs, global_ins.repeat(1, inputs.size(1), 1)], dim=-1)
        r = torch.sigmoid(self.gru_ir(combined_inputs) + self.gru_hr(agg_msgs))
        i = torch.sigmoid(self.gru_ii(combined_inputs) + self.gru_hi(agg_msgs))
        n = torch.tanh(self.gru_in(combined_inputs) + r * self.gru_hn(agg_msgs))

        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob, training=self.training)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob, training=self.training)
        pred = self.out_fc3(pred)

        # Do a skip connection
        assert inputs.shape == pred.shape, "Input feature dim should match output feature dim"
        pred = inputs + pred

        return pred, hidden

    def forward(
        self,
        inputs,
        global_inputs,
        rel_rec,
        rel_send,
        rel_types,
        burn_in,
        burn_in_steps,
        split_len,
    ):
        # Inputs should be [B, T, N, F]

        pred_all = []

        hidden = Variable(
            torch.zeros(inputs.size(0), inputs.size(2), self.gru_hid, device=inputs.device)
        )

        for step in range(0, inputs.shape[1] - 1):
            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, step, :, :]
                    global_ins = global_inputs[:, step, :]
                else:
                    ins = pred_all[step - 1]
                    global_ins = global_inputs[:, step, :]

            pred, hidden = self.do_single_step_forward(
                ins, global_ins, rel_rec, rel_send, rel_types, hidden
            )
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds