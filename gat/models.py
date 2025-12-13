import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv as PyGGATConv
from torch_geometric.utils import softmax


class Bias(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.Tensor(size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x + self.bias


class GATConv(nn.Module):
    """
    GAT Convolution layer using PyTorch Geometric with hybrid approach.
    Uses PyG's GATConv as base and adds custom symmetric normalization.
    """
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        norm="none",
    ):
        super(GATConv, self).__init__()
        if norm not in ("none", "both"):
            raise ValueError('Invalid norm value. Must be either "none", "both". But got "{}".'.format(norm))

        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._norm = norm

        # Use PyTorch Geometric's GATConv
        self.gat_conv = PyGGATConv(
            in_feats,
            out_feats,
            heads=num_heads,
            concat=True,  # Concatenate heads
            negative_slope=negative_slope,
            dropout=attn_drop,
            add_self_loops=False,  # We handle self-loops externally
            bias=False,
        )

        self.feat_drop = nn.Dropout(feat_drop)

        # Residual connection
        if residual:
            if in_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = None

        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        self.gat_conv.reset_parameters()
        if isinstance(self.res_fc, nn.Linear):
            gain = nn.init.calculate_gain("relu")
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_feats]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Output features [num_nodes, num_heads, out_feats]
        """
        h = self.feat_drop(x)

        # Apply symmetric normalization if needed (before message passing)
        if self._norm == "both":
            # Compute out-degree normalization D^{-0.5}
            row, col = edge_index
            deg = torch.zeros(x.size(0), device=x.device)
            deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
            deg = deg.clamp(min=1)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            h = h * deg_inv_sqrt.view(-1, 1)

        # Apply GAT convolution
        out = self.gat_conv(h, edge_index)

        # Reshape to [num_nodes, num_heads, out_feats]
        out = out.view(-1, self._num_heads, self._out_feats)

        # Apply symmetric normalization if needed (after message passing)
        if self._norm == "both":
            # Compute in-degree normalization D^{0.5}
            row, col = edge_index
            deg = torch.zeros(x.size(0), device=x.device)
            deg.scatter_add_(0, col, torch.ones(col.size(0), device=x.device))
            deg = deg.clamp(min=1)
            deg_sqrt = deg.pow(0.5)
            # Reshape for broadcasting: [num_nodes, 1, 1]
            out = out * deg_sqrt.view(-1, 1, 1)

        # Add residual connection
        if self.res_fc is not None:
            if isinstance(self.res_fc, nn.Identity):
                res = x.view(x.shape[0], self._num_heads, self._out_feats)
            else:
                res = self.res_fc(x).view(x.shape[0], self._num_heads, self._out_feats)
            out = out + res

        # Apply activation
        if self._activation is not None:
            out = self._activation(out)

        return out


class GAT(nn.Module):
    def __init__(
        self, in_feats, n_classes, n_hidden, n_layers, n_heads, activation, dropout=0.0, attn_drop=0.0, norm="none"
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            out_channels = n_heads

            self.convs.append(GATConv(in_hidden, out_hidden, num_heads=n_heads, attn_drop=attn_drop, norm=norm))

            self.linear.append(nn.Linear(in_hidden, out_channels * out_hidden, bias=False))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels * out_hidden))

        self.bias_last = Bias(n_classes)

        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_feats]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Class logits [num_nodes, n_classes]
        """
        h = x
        h = self.dropout0(h)

        for i in range(self.n_layers):
            conv = self.convs[i](h, edge_index, edge_weight)
            linear = self.linear[i](h).view(conv.shape)

            h = conv + linear

            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        h = h.mean(1)
        h = self.bias_last(h)

        return h
