import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model_utils.network_util import (MLP, Aggre_Index, Gen_Index,
                                                build_mlp)
from src.model.transformer.attention import MultiHeadAttention


class GraphEdgeAttenNetwork(torch.nn.Module):
    def __init__(self, num_heads, dim_node, dim_edge, dim_atten, aggr='max', use_bn=False,
                 flow='target_to_source', attention='fat', use_edge:bool=True, **kwargs):
        super().__init__()
        self.name = 'edgeatten'
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.index_get = Gen_Index(flow=flow)
        if attention == 'fat':        
            self.index_aggr = Aggre_Index(aggr=aggr, flow=flow)
        elif attention == 'distance':
            aggr = 'add'
            self.index_aggr = Aggre_Index(aggr=aggr, flow=flow)
        else:
            raise NotImplementedError()

        self.edge_gate = nn.Sequential(
            nn.Linear(dim_edge, dim_edge // 2),
            nn.ReLU(),
            nn.BatchNorm1d(dim_edge // 2),
            nn.Linear(dim_edge // 2, 1),
            nn.Sigmoid()
        )
        
        self.edgeatten = MultiHeadedEdgeAttention(
            dim_node=dim_node, dim_edge=dim_edge, dim_atten=dim_atten,
            num_heads=num_heads, use_bn=use_bn, attention=attention, use_edge=use_edge, **kwargs)
        
        self.prop = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                            do_bn=use_bn, on_last=False)
        self.layer_norm = nn.LayerNorm(dim_node)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_feature, edge_index, weight=None, istrain=False):
        assert x.ndim == 2
        assert edge_feature.ndim == 2
        x_i, x_j = self.index_get(x, edge_index)
        
        edge_dict = {}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_dict[(src, dst)] = i
        
        reverse_edge_feature = torch.zeros_like(edge_feature)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if (dst, src) in edge_dict:
                reverse_idx = edge_dict[(dst, src)]
                reverse_edge_feature[i] = edge_feature[reverse_idx]
        
        gates = self.edge_gate(edge_feature)
        reverse_edge_feature = gates * reverse_edge_feature
        
        xx, gcn_edge_feature, prob = self.edgeatten(x_i, edge_feature, reverse_edge_feature, x_j, weight, istrain=istrain)
        
        subject_edges = {}
        object_edges = {}
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            
            if src not in subject_edges:
                subject_edges[src] = []
            subject_edges[src].append(i)
            
            if dst not in object_edges:
                object_edges[dst] = []
            object_edges[dst].append(i)
        
        xx = self.index_aggr(xx, edge_index, dim_size=x.shape[0])
        
        bi_edge_attention = torch.zeros_like(xx)
        for node_idx in range(x.shape[0]):
            subj_features = []
            if node_idx in subject_edges:
                for edge_idx in subject_edges[node_idx]:
                    subj_features.append(gcn_edge_feature[edge_idx])
            
            obj_features = []
            if node_idx in object_edges:
                for edge_idx in object_edges[node_idx]:
                    obj_features.append(gcn_edge_feature[edge_idx])
            
            if subj_features:
                subj_agg = torch.stack(subj_features).mean(dim=0)
            else:
                subj_agg = torch.zeros(self.dim_edge, device=x.device)
                
            if obj_features:
                obj_agg = torch.stack(obj_features).mean(dim=0)
            else:
                obj_agg = torch.zeros(self.dim_edge, device=x.device)
            
            edge_agg = torch.cat([subj_agg, obj_agg])
            
            bi_edge_attention[node_idx] = nn.Linear(edge_agg.shape[0], xx.shape[1], device=x.device)(edge_agg)
        
        xx = F.relu(xx) * self.sigmoid(bi_edge_attention)
        
        xx = self.prop(torch.cat([x, xx], dim=1))
        xx = self.layer_norm(xx)
        
        return xx, gcn_edge_feature


class MultiHeadedEdgeAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_node: int, dim_edge: int, dim_atten: int, use_bn=False,
                 attention='fat', use_edge:bool=True, **kwargs):
        super().__init__()
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.name = 'MultiHeadedEdgeAttention'
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.d_n = d_n = dim_node // num_heads
        self.d_e = d_e = dim_edge // num_heads
        self.d_o = d_o = dim_atten // num_heads
        self.num_heads = num_heads
        self.use_edge = use_edge
        
        self.nn_edge = build_mlp([dim_node*2+dim_edge*2, (dim_node+dim_edge*2), dim_edge],
                          do_bn=use_bn, on_last=False)
        self.edge_layer_norm = nn.LayerNorm(dim_edge)
        self.mask_obj = 0.5
        
        DROP_OUT_ATTEN = kwargs.get('DROP_OUT_ATTEN', 0.5)  
        
        self.attention = attention
        assert self.attention in ['fat']
        
        if self.attention == 'fat':
            if use_edge:
                self.nn = MLP([d_n+d_e, d_n+d_e, d_o], do_bn=use_bn, drop_out=DROP_OUT_ATTEN)
            else:
                self.nn = MLP([d_n, d_n*2, d_o], do_bn=use_bn, drop_out=DROP_OUT_ATTEN)
                
            self.proj_edge = build_mlp([dim_edge, dim_edge])
            self.proj_query = build_mlp([dim_node, dim_node])
            self.proj_value = build_mlp([dim_node, dim_atten])
        elif self.attention == 'distance':
            self.proj_value = build_mlp([dim_node, dim_atten])

        
    def forward(self, query, edge, reverse_edge, value, weight=None, istrain=False):
        batch_dim = query.size(0)
        
        edge_feature = torch.cat([query, edge, reverse_edge, value], dim=1)
        edge_feature = self.nn_edge(edge_feature)
        edge_feature = self.edge_layer_norm(edge_feature)

        if self.attention == 'fat':
            value = self.proj_value(value)
            query = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads)
            edge = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)
            if self.use_edge:
                prob = self.nn(torch.cat([query, edge], dim=1))  # b, dim, head    
            else:
                prob = self.nn(query)  # b, dim, head
                
            prob = prob.softmax(1)
            x = torch.einsum('bm,bm->bm', prob.reshape_as(value), value)
        
        elif self.attention == 'distance':
            raise NotImplementedError()
        
        else:
            raise NotImplementedError('')
        
        return x, edge_feature, prob
    

class MMG_pt_single(torch.nn.Module):
    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr='max', 
                 use_bn=False, flow='target_to_source', attention='fat', 
                 hidden_size=512, depth=1, use_edge:bool=True, **kwargs):
        
        super().__init__()

        self.num_heads = num_heads
        depth = 5
        self.depth = depth

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads) 
            for i in range(depth)
        )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim_node) for _ in range(depth)
        ])

        self.gcn_3ds = torch.nn.ModuleList()
        
        for _ in range(self.depth):
            self.gcn_3ds.append(TripletGCN(dim_node=dim_node, dim_edge=dim_edge, dim_hidden=hidden_size))
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
        self.self_attn_fc = nn.Sequential(  # 11 32 32 4(head)
            nn.Linear(4, 32),  # xyz, dist
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_heads)
        )
        
        self.residual_proj = nn.ModuleList([
            nn.Linear(dim_node, dim_node) for _ in range(depth)
        ])
    
    def forward(self, obj_feature_3d, edge_feature_3d, edge_index, batch_ids, obj_center=None, istrain=False):
        
        for i in range(self.depth):
            obj_feature_3d, edge_feature_3d = self.gcn_3ds[i](obj_feature_3d, edge_feature_3d, edge_index)
            
            if i < (self.depth-1) or self.depth==1:
                obj_feature_3d = F.relu(obj_feature_3d)
                obj_feature_3d = self.drop_out(obj_feature_3d)
                
                edge_feature_3d = F.relu(edge_feature_3d)
                edge_feature_3d = self.drop_out(edge_feature_3d)
        
        return obj_feature_3d, edge_feature_3d

from typing import Optional
import torch
from torch import Tensor
from src.model.model_utils.networks_base import BaseNetwork, mySequential
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
    
class TripletGCN(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr= 'add', use_bn=True):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_hidden = dim_hidden
        self.nn1 = build_mlp([dim_node*2+dim_edge*2, dim_hidden, dim_hidden*2+dim_edge],
                      do_bn= use_bn, on_last=True)
        self.nn2 = build_mlp([dim_hidden,dim_hidden,dim_node],do_bn= use_bn)
        self.index_get = Gen_Index(flow='target_to_source')
        self.edge_gate = nn.Sequential(
            nn.Linear(dim_edge, dim_edge // 2),
            nn.ReLU(),
            nn.BatchNorm1d(dim_edge // 2),
            nn.Linear(dim_edge // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_feature, edge_index):
        edge_dict = {}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_dict[(src, dst)] = i
        
        reverse_edge_feature = torch.zeros_like(edge_feature)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if (dst, src) in edge_dict:
                reverse_idx = edge_dict[(dst, src)]
                reverse_edge_feature[i] = edge_feature[reverse_idx]
        
        gates = self.edge_gate(edge_feature)
        reverse_edge_feature = gates * reverse_edge_feature
        gcn_x, gcn_e = self.propagate(edge_index, x=x, edge_feature=edge_feature, reverse_edge_feature=reverse_edge_feature)
        gcn_x = x + self.nn2(gcn_x)
        return gcn_x, gcn_e

    def message(self, x_i, x_j,edge_feature, reverse_edge_feature):
        x = torch.cat([x_i,edge_feature,reverse_edge_feature,x_j],dim=1)
        x = self.nn1(x)#.view(b,-1)
        new_x_i = x[:,:self.dim_hidden]
        new_e   = x[:,self.dim_hidden:(self.dim_hidden+self.dim_edge)]
        new_x_j = x[:,(self.dim_hidden+self.dim_edge):]
        x = new_x_i+new_x_j
        return [x, new_e]
    
    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x


class TripletGCNModel(BaseNetwork):
    """ A sequence of scene graph convolution layers  """
    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = torch.nn.ModuleList()
        
        for _ in range(self.num_layers):
            self.gconvs.append(TripletGCN(**kwargs))

    def forward(self, node_feature, edge_feature, edges_indices):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature = gconv(node_feature, edge_feature, edges_indices)
            
            if i < (self.num_layers-1):
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)
        return node_feature, edge_feature