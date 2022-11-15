from torch import nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class MLP(nn.Module):
    """The standard MLP module w. batchnorm, initializationa and dropout"""

    def __init__(self, n_in, n_hid, n_out, dropout_prob=0, use_bn=True):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = dropout_prob
        self.use_bn = use_bn

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        """We do batch norm over batches and things so we reshape first"""
        # x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        orig_shape = inputs.shape
        x = inputs.view(-1, inputs.size(-1))
        x = self.bn(x)
        return x.view(orig_shape)

    def forward(self, inputs):
        # input shape [num_sims / batches, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        if self.use_bn:
            x = self.batch_norm(x)
        return x


class MPNN(nn.Module):
    """Simple non recurrent decoder, based on NRI paper."""

    def __init__(self, n_in, n_hid, n_out, msg_hid, msg_out, n_edge_types, dropout_prob):
        super().__init__()

        self.n_edge_types = n_edge_types
        self.dropout_prob = dropout_prob

        # FC layers to compute messages
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(in_features=n_in * 2, out_features=msg_hid) for _ in range(n_edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(in_features=msg_hid, out_features=msg_out) for _ in range(n_edge_types)]
        )

        # FC for generating the output
        self.out_fc1 = nn.Linear(in_features=msg_out + n_in, out_features=n_hid)
        self.out_fc2 = nn.Linear(in_features=n_hid, out_features=n_hid)
        self.out_fc3 = nn.Linear(in_features=n_hid, out_features=n_out)

        self.msg_out_shape = msg_out

    def edge2node(self, x, rel_rec):
        """This function makes the aggregation over the incomming edge embeddings"""
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        """This function makes a matrix of [node_i, node_j] rows for the edge embeddings"""
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send, rel_types):
        """[summary]"""

        # input shape [batch_size, num_atoms, num_dims]

        pre_msg = self.node2edge(inputs, rel_rec, rel_send)

        # Create variable to aggregate the messages in
        all_msgs = Variable(
            torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape, device=inputs.device)
        )

        # Go over the different edge types and compute their contribution to the overall messages
        for i in range(0, self.n_edge_types):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * rel_types[:, :, i : i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_msgs = torch.cat([inputs, agg_msgs], dim=-1)

        # Output MLP
        output = F.dropout(
            F.relu(self.out_fc1(aug_msgs)), p=self.dropout_prob, training=self.training
        )
        output = F.dropout(F.relu(self.out_fc2(output)), p=self.dropout_prob, training=self.training)
        output = self.out_fc3(output)

        return output