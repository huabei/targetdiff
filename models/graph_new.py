import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear, Conv1d, ModuleList
from torch_scatter import scatter_sum, scatter_softmax
from torch_geometric.nn import radius_graph, knn_graph
# from models.common import GaussianSmearing, MLP, NONLINEARITIES


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    # "swish": Swish(),
    'silu': nn.SiLU()
}


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50, type_='exp'):
        super().__init__()
        self.start = start
        self.stop = stop
        if type_ == 'exp':
            offset = torch.exp(torch.linspace(start=np.log(start+1), end=np.log(stop+1), steps=num_gaussians)) - 1
        elif type_ == 'linear':
            offset = torch.linspace(start=start, end=stop, steps=num_gaussians)
        else:
            raise NotImplementedError('type_ must be either exp or linear')
        diff = torch.diff(offset)
        diff = torch.cat([diff[:1], diff])
        coeff = -0.5 / (diff**2)
        self.register_buffer('coeff', coeff)
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.clamp_min(self.start)
        dist = dist.clamp_max(self.stop)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class NodeBlock(Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate):
        super().__init__()
        self.use_gate = use_gate
        self.node_dim = node_dim
        
        self.node_net = MLP(node_dim, hidden_dim, hidden_dim)
        self.edge_net = MLP(edge_dim, hidden_dim, hidden_dim)
        self.msg_net = Linear(hidden_dim, hidden_dim)

        if self.use_gate:
            self.gate = MLP(edge_dim+node_dim+1, hidden_dim, hidden_dim) # add 1 for time

        self.centroid_lin = Linear(node_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.out_transform = Linear(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr, node_time):
        """
        Args:
            x:  Node features, (N, H).
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        N = x.size(0)
        row, col = edge_index   # (E,) , (E,)

        h_node = self.node_net(x)  # (N, H)

        # Compose messages
        h_edge = self.edge_net(edge_attr)  # (E, H_per_head)
        msg_j = self.msg_net(h_edge * h_node[col])

        if self.use_gate:
            gate = self.gate(torch.cat([edge_attr, x[col], node_time[col]], dim=-1))
            msg_j = msg_j * torch.sigmoid(gate)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N)
        out = self.centroid_lin(x) + aggr_msg

        out = self.layer_norm(out)
        out = self.out_transform(self.act(out))
        return out


class BondFFN(Module):
    def __init__(self, bond_dim, node_dim, inter_dim, use_gate, out_dim=None):
        super().__init__()
        out_dim = bond_dim if out_dim is None else out_dim
        self.use_gate = use_gate
        self.bond_linear = Linear(bond_dim, inter_dim, bias=False)
        self.node_linear = Linear(node_dim, inter_dim, bias=False)
        self.inter_module = MLP(inter_dim, out_dim, inter_dim)
        if self.use_gate:
            self.gate = MLP(bond_dim+node_dim+1, out_dim, 32)  # +1 for time

    def forward(self, bond_feat_input, node_feat_input, time):
        bond_feat = self.bond_linear(bond_feat_input)
        node_feat = self.node_linear(node_feat_input)
        inter_feat = bond_feat * node_feat
        inter_feat = self.inter_module(inter_feat)
        if self.use_gate:
            gate = self.gate(torch.cat([bond_feat_input, node_feat_input, time], dim=-1))
            inter_feat = inter_feat * torch.sigmoid(gate)
        return inter_feat


class EdgeBlock(Module):
    def __init__(self, edge_dim, node_dim, hidden_dim=None, use_gate=True):
        super().__init__()
        self.use_gate = use_gate
        inter_dim = edge_dim * 2 if hidden_dim is None else hidden_dim

        self.bond_ffn_left = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        self.bond_ffn_right = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)

        self.node_ffn_left = Linear(node_dim, edge_dim)
        self.node_ffn_right = Linear(node_dim, edge_dim)

        self.self_ffn = Linear(edge_dim, edge_dim)
        self.layer_norm = nn.LayerNorm(edge_dim)
        self.out_transform = Linear(edge_dim, edge_dim)
        self.act = nn.ReLU()

    def forward(self, h_bond, bond_index, h_node, bond_time):
        """
        h_bond: (b, bond_dim)
        bond_index: (2, b)
        h_node: (n, node_dim)
        """
        N = h_node.size(0)
        left_node, right_node = bond_index

        # message from neighbor bonds
        msg_bond_left = self.bond_ffn_left(h_bond, h_node[left_node], bond_time)
        msg_bond_left = scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        msg_bond_left = msg_bond_left[left_node]

        msg_bond_right = self.bond_ffn_right(h_bond, h_node[right_node], bond_time)
        msg_bond_right = scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        msg_bond_right = msg_bond_right[right_node]
        
        h_bond = (
            msg_bond_left + msg_bond_right
            + self.node_ffn_left(h_node[left_node])
            + self.node_ffn_right(h_node[right_node])
            + self.self_ffn(h_bond)
        )
        h_bond = self.layer_norm(h_bond)

        h_bond = self.out_transform(self.act(h_bond))
        return h_bond


class NodeEdgeNet(Module):
    def __init__(self, node_dim, edge_dim, num_blocks, cutoff, use_gate, **kwargs):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_blocks = num_blocks
        self.cutoff = cutoff
        self.use_gate = use_gate
        self.kwargs = kwargs

        if 'num_gaussians' not in kwargs:
            num_gaussians = 16
        else:
            num_gaussians = kwargs['num_gaussians']
        if 'start' not in kwargs:
            start = 0
        else:
            start = kwargs['start']
        self.distance_expansion = GaussianSmearing(start=start, stop=cutoff, num_gaussians=num_gaussians)
        if ('update_edge' in kwargs) and (not kwargs['update_edge']):
            self.update_edge = False
            input_edge_dim = num_gaussians
        else:
            self.update_edge = True  # default update edge
            input_edge_dim = edge_dim + num_gaussians
            
        if ('update_pos' in kwargs) and (not kwargs['update_pos']):
            self.update_pos = False
        else:
            self.update_pos = True  # default update pos
        
        # node network
        self.node_blocks_with_edge = ModuleList()
        self.node_blocks_with_edge_2 = ModuleList()
        self.edge_embs = ModuleList()
        self.edge_blocks = ModuleList()
        self.pos_blocks = ModuleList()
        self.pos_blocks_2 = ModuleList()
        for _ in range(num_blocks):
            self.node_blocks_with_edge.append(NodeBlock(
                node_dim=node_dim, edge_dim=edge_dim, hidden_dim=node_dim, use_gate=use_gate,
            ))
            self.node_blocks_with_edge_2.append(NodeBlock(
                node_dim=node_dim, edge_dim=edge_dim, hidden_dim=node_dim, use_gate=use_gate,
            ))
            
            self.edge_embs.append(Linear(input_edge_dim, edge_dim))
            if self.update_edge:
                self.edge_blocks.append(EdgeBlock(
                    edge_dim=edge_dim, node_dim=node_dim, use_gate=use_gate,
                ))
            if self.update_pos:
                self.pos_blocks.append(PosUpdate(
                    node_dim=node_dim, edge_dim=edge_dim, hidden_dim=edge_dim, use_gate=use_gate,
                ))
                self.pos_blocks_2.append(PosUpdate(
                    node_dim=node_dim, edge_dim=edge_dim, hidden_dim=edge_dim, use_gate=use_gate,
                ))
                
    def forward(self, h_node, pos_node, h_edge, edge_index, node_time, edge_time):
        # 后1/3的edge翻转
        edge_num = edge_index.shape[1]
        for i in range(self.num_blocks):
            # edge fetures before each block
            if self.update_pos or (i==0):
                h_edge_dist, relative_vec, distance = self._build_edges_dist(pos_node, edge_index)
            if self.update_edge:
                h_edge = torch.cat([h_edge, h_edge_dist], dim=-1)
            else:
                h_edge = h_edge_dist
            h_edge = self.edge_embs[i](h_edge)

            # node and edge feature updates
            h_node_with_edge = self.node_blocks_with_edge[i](h_node, edge_index[:, :edge_num//2], h_edge[:edge_num//2], node_time)
            # 双向更新
            h_node_with_edge_2 = self.node_blocks_with_edge_2[i](h_node, edge_index[:, edge_num//2:], h_edge[edge_num//2:], node_time)
            # TODO: 创建pocket指向ligand的键，增加pocket对ligand节点的更新
            
            if self.update_edge:
                h_edge = h_edge + self.edge_blocks[i](h_edge, edge_index, h_node, edge_time)
            # print(h_node_with_edge.shape, h_node_with_edge_2.shape)
            h_node = h_node + h_node_with_edge + h_node_with_edge_2
            # pos updates
            if self.update_pos:
                # pos_node = pos_node + self.pos_blocks[i](h_node, h_edge, edge_index, relative_vec, distance, edge_time)
                pos_node = (pos_node +
                            self.pos_blocks[i](h_node, h_edge[:edge_num//2], edge_index[:, :edge_num//2], relative_vec[:edge_num//2], distance[:edge_num//2], edge_time[:edge_num//2]) +
                            self.pos_blocks_2[i](h_node, h_edge[edge_num//2:], edge_index[:, edge_num//2:], relative_vec[edge_num//2:], distance[edge_num//2:], edge_time[edge_num//2:])
                            )
        return h_node, pos_node, h_edge

    def _build_edges_dist(self, pos, edge_index):
        # distance
        relative_vec = pos[edge_index[0]] - pos[edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        edge_dist = self.distance_expansion(distance)
        return edge_dist, relative_vec, distance
    


class PosUpdate(Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate):
        super().__init__()
        self.left_lin_edge = MLP(node_dim, edge_dim, hidden_dim)
        self.right_lin_edge = MLP(node_dim, edge_dim, hidden_dim)
        self.edge_lin = BondFFN(edge_dim, edge_dim, node_dim, use_gate, out_dim=1)

    def forward(self, h_node, h_edge, edge_index, relative_vec, distance, edge_time):
        edge_index_left, edge_index_right = edge_index
        
        left_feat = self.left_lin_edge(h_node[edge_index_left])
        right_feat = self.right_lin_edge(h_node[edge_index_right])
        weight_edge = self.edge_lin(h_edge, left_feat * right_feat, edge_time)
        
        # relative_vec = pos_node[edge_index_left] - pos_node[edge_index_right]
        # distance = torch.norm(relative_vec, dim=-1, keepdim=True)
        force_edge = weight_edge * relative_vec / distance.unsqueeze(-1) / (distance.unsqueeze(-1) + 1.)
        delta_pos = scatter_sum(force_edge, edge_index_left, dim=0, dim_size=h_node.shape[0])

        return delta_pos
