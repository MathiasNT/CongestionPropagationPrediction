import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


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

    def edge2node(self, x, rel_rec):
        """This function makes the aggregation over the incomming edge embeddings"""
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        """This function makes a matrix of [node_i, node_j] rows for the edge embeddings"""
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=-1)
        return edges

    def do_single_step_forward(self, inputs, rel_rec, rel_send, rel_types, hidden):

        # input shape [batch_size, num_timesteps, num_atoms, num_dims]
        # rel_types [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]
        pre_msg = self.node2edge(hidden, rel_rec, rel_send)

        # Create variable to aggregate the messages in
        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape, device=inputs.device))

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
    ):
        # Inputs should be [B, T, N, F]
        inputs = inputs.permute(0, 2, 1, 3)
        pred_all = []

        hidden = Variable(torch.zeros(inputs.size(0), inputs.size(2), self.gru_hid, device=inputs.device))

        for step in range(0, inputs.shape[1] - 1):
            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            pred, hidden = self.do_single_step_forward(ins, rel_rec, rel_send, rel_types, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds, hidden
