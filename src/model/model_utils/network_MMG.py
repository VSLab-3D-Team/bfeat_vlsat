import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model_utils.network_util import (MLP, Aggre_Index, Gen_Index,
                                                build_mlp)
from src.model.transformer.attention import MultiHeadAttention


class GraphEdgeAttenNetwork(torch.nn.Module):
    def __init__(self, num_heads, dim_node, dim_edge, dim_atten, aggr='max', use_bn=False,
                 flow='target_to_source', attention='fat', use_edge:bool=True, 
                 use_bidirectional:bool=True, distance_sigma:float=1.0, **kwargs):
        super().__init__()
        self.name = 'edgeatten'
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.use_bidirectional = use_bidirectional
        self.distance_sigma = distance_sigma
        self.index_get = Gen_Index(flow=flow)
        
        if attention == 'fat':        
            self.index_aggr = Aggre_Index(aggr=aggr, flow=flow)
        elif attention == 'distance':
            aggr = 'add'
            self.index_aggr = Aggre_Index(aggr=aggr, flow=flow)
        else:
            raise NotImplementedError()

        self.edgeatten = MultiHeadedEdgeAttention(
            dim_node=dim_node, dim_edge=dim_edge, dim_atten=dim_atten,
            num_heads=num_heads, use_bn=use_bn, attention=attention, 
            use_edge=use_edge, use_bidirectional=use_bidirectional,
            distance_sigma=distance_sigma, **kwargs)
        
        if use_bidirectional:
            self.prop = build_mlp([dim_node+dim_atten*2, dim_node+dim_atten, dim_node],
                                do_bn=use_bn, on_last=False)
        else:
            self.prop = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                                do_bn=use_bn, on_last=False)

    def forward(self, x, edge_feature, edge_index, weight=None, istrain=False, obj_center=None):
        assert x.ndim == 2
        assert edge_feature.ndim == 2
        
        x_i, x_j = self.index_get(x, edge_index)
        
        if self.use_bidirectional:
            edge_index_reverse = torch.stack([edge_index[1], edge_index[0]], dim=0)
            
            reverse_edge_dict = {}
            for idx, (src, dst) in enumerate(edge_index.t().tolist()):
                reverse_edge_dict[(src, dst)] = idx
            
            reverse_edge_feature = torch.zeros_like(edge_feature)
            for idx, (src, dst) in enumerate(edge_index.t().tolist()):
                if (dst, src) in reverse_edge_dict:
                    reverse_idx = reverse_edge_dict[(dst, src)]
                    reverse_edge_feature[idx] = edge_feature[reverse_idx]
            
            obj_centers = None
            if obj_center is not None:
                src_centers = obj_center[edge_index[0]]
                dst_centers = obj_center[edge_index[1]]
                obj_centers = (src_centers, dst_centers)
            
            xx, gcn_edge_feature, prob = self.edgeatten(
                x_i, edge_feature, x_j, reverse_edge=reverse_edge_feature, 
                weight=weight, istrain=istrain, obj_centers=obj_centers
            )
            
            xx_in = self.index_aggr(xx, edge_index, dim_size=x.shape[0])
            
            xx_out = self.index_aggr(xx, edge_index_reverse, dim_size=x.shape[0])
            
            xx_combined = self.prop(torch.cat([x, xx_in, xx_out], dim=1)) # bidirectional node update
            
            return xx_combined, gcn_edge_feature
        else:
            obj_centers = None
            if obj_center is not None:
                src_centers = obj_center[edge_index[0]]
                dst_centers = obj_center[edge_index[1]]
                obj_centers = (src_centers, dst_centers)
                
            xx, gcn_edge_feature, prob = self.edgeatten(
                x_i, edge_feature, x_j, weight=weight, istrain=istrain, obj_centers=obj_centers
            )
            
            xx = self.index_aggr(xx, edge_index, dim_size=x.shape[0])
            xx = self.prop(torch.cat([x, xx], dim=1))
            
            return xx, gcn_edge_feature
  

class MultiHeadedEdgeAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_node: int, dim_edge: int, dim_atten: int, use_bn=False,
                 attention='fat', use_edge:bool=True, use_bidirectional:bool=True, 
                 distance_sigma:float=1.0, **kwargs):
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
        self.use_bidirectional = use_bidirectional
        self.distance_sigma = distance_sigma
        
        if self.use_bidirectional:
            self.nn_edge = build_mlp([dim_node*2+dim_edge*2, (dim_node+dim_edge*2), dim_edge],
                              do_bn=use_bn, on_last=False)
        else:
            self.nn_edge = build_mlp([dim_node*2+dim_edge, (dim_node+dim_edge), dim_edge],
                              do_bn=use_bn, on_last=False)
        
        self.mask_obj = 0.5
        
        DROP_OUT_ATTEN = None
        if 'DROP_OUT_ATTEN' in kwargs:
            DROP_OUT_ATTEN = kwargs['DROP_OUT_ATTEN']
        
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
    
    def forward(self, query, edge, value, reverse_edge=None, weight=None, istrain=False, obj_centers=None):
        batch_dim = query.size(0)
        
        if self.use_bidirectional and reverse_edge is not None:
            edge_feature = torch.cat([query, edge, reverse_edge, value], dim=1)
        else:
            edge_feature = torch.cat([query, edge, value], dim=1)
        
        if obj_centers is not None:
            src_centers, dst_centers = obj_centers
            
            distances = torch.norm(src_centers - dst_centers, dim=1, keepdim=True)  # [batch_size, 1]
            
            edge_distance_weights = torch.exp(-(distances ** 2) / (2 * self.distance_sigma ** 2))  # [batch_size, 1]
            
            edge_feature = self.nn_edge(edge_feature) * edge_distance_weights
        else:
            edge_feature = self.nn_edge(edge_feature)
        
        if self.attention == 'fat':
            value_proj = self.proj_value(value)
            query = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads)
            edge = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)
            
            if self.use_edge:
                prob = self.nn(torch.cat([query, edge], dim=1))  # b, dim, head    
            else:
                prob = self.nn(query)  # b, dim, head 
            
            prob = prob.softmax(1)
            
            if obj_centers is not None:
                node_distance_weights = torch.exp(-(distances ** 2) / (2 * self.distance_sigma ** 2))  # [batch_size, 1]
                
                expanded_weights = node_distance_weights.unsqueeze(-1)
                combined_weights = prob * expanded_weights
                combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-10)
                weights_avg = combined_weights.mean(dim=2)  # [batch_size, dim]
                
                x = torch.sum(weights_avg.unsqueeze(2) * value_proj.unsqueeze(1), dim=1)
            else:
                weights_avg = prob.mean(dim=2)  # [batch_size, dim]
                
                x = torch.sum(weights_avg.unsqueeze(2) * value_proj.unsqueeze(1), dim=1)
        
        elif self.attention == 'distance':
            raise NotImplementedError()
        
        else:
            raise NotImplementedError('')
        
        return x, edge_feature, prob
    

class MMG_single(torch.nn.Module):
    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr='max', 
                 use_bn=False, flow='target_to_source', attention='fat', 
                 hidden_size=512, depth=1, use_edge:bool=True, 
                 use_bidirectional:bool=True, distance_sigma:float=1.0, **kwargs):
        
        super().__init__()
        self.num_heads = num_heads
        self.depth = depth
        self.gcn_3ds = torch.nn.ModuleList()
        
        for _ in range(self.depth):
            self.gcn_3ds.append(GraphEdgeAttenNetwork(
                num_heads,
                dim_node,
                dim_edge,
                dim_atten,
                aggr,
                use_bn=use_bn,
                flow=flow,
                attention=attention,
                use_edge=use_edge,
                use_bidirectional=use_bidirectional,
                distance_sigma=distance_sigma,
                **kwargs))
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
    
    def forward(self, obj_feature_3d, edge_feature_3d, edge_index, batch_ids, obj_center=None, istrain=False):
        for i in range(self.depth):
            obj_feature_3d, edge_feature_3d = self.gcn_3ds[i](
                obj_feature_3d, edge_feature_3d, edge_index, istrain=istrain, obj_center=obj_center
            )
            
            if i < (self.depth-1) or self.depth==1:
                obj_feature_3d = F.relu(obj_feature_3d)
                obj_feature_3d = self.drop_out(obj_feature_3d)
                
                edge_feature_3d = F.relu(edge_feature_3d)
                edge_feature_3d = self.drop_out(edge_feature_3d)
        
        return obj_feature_3d, edge_feature_3d