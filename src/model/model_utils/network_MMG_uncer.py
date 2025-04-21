import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model_utils.network_util import (MLP, Aggre_Index, Gen_Index,
                                                build_mlp)
from src.model.transformer.attention import MultiHeadAttention


class GraphEdgeAttenNetwork(torch.nn.Module):
    def __init__(self, num_heads, dim_node, dim_edge, dim_atten, aggr= 'max', use_bn=False,
                 flow='target_to_source',attention = 'fat',use_edge:bool=True, **kwargs):
        super().__init__()
        self.name = 'edgeatten'
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.index_get = Gen_Index(flow=flow)
        if attention == 'fat':        
            self.index_aggr = Aggre_Index(aggr=aggr,flow=flow)
        elif attention == 'distance':
            aggr = 'add'
            self.index_aggr = Aggre_Index(aggr=aggr,flow=flow)
        else:
            raise NotImplementedError()

        self.edgeatten = MultiHeadedEdgeAttention(
            dim_node=dim_node,dim_edge=dim_edge,dim_atten=dim_atten,
            num_heads=num_heads,use_bn=use_bn,attention=attention,use_edge=use_edge, **kwargs)
        self.prop = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                            do_bn= use_bn, on_last=False)

    def forward(self, x, edge_feature, edge_index, weight=None, istrain=False):
        assert x.ndim == 2
        assert edge_feature.ndim == 2
        x_i, x_j = self.index_get(x, edge_index)
        xx, gcn_edge_feature, prob, uncertainty = self.edgeatten(x_i, edge_feature, x_j, weight, istrain=istrain)
        xx = self.index_aggr(xx, edge_index, dim_size = x.shape[0])
        xx = self.prop(torch.cat([x,xx],dim=1))
        return xx, gcn_edge_feature, uncertainty
  

class DualUncertaintyEstimator(nn.Module):
    def __init__(self, dim_node):
        super().__init__()
        self.aleatoric_estimator = nn.Sequential(
            nn.Linear(dim_node, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.epistemic_estimator = nn.Sequential(
            nn.Linear(dim_node, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        aleatoric = self.aleatoric_estimator(x)
        epistemic = self.epistemic_estimator(x)
        combined = torch.cat([aleatoric, epistemic], dim=1)
        return aleatoric, epistemic, combined


class UncertaintyPropagation(nn.Module):
    def __init__(self, dim_edge, num_heads):
        super().__init__()
        self.edge_uncertainty = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads),
            nn.Sigmoid()
        )
    
    def forward(self, src_uncertainty, tgt_uncertainty):
        combined_uncertainty = torch.cat([src_uncertainty, tgt_uncertainty], dim=1)
        edge_uncertainty = self.edge_uncertainty(combined_uncertainty)
        return edge_uncertainty


class AdaptiveTemperatureAttention(nn.Module):
    def __init__(self, dim_uncertainty, num_heads):
        super().__init__()
        self.temp_predictor = nn.Sequential(
            nn.Linear(dim_uncertainty, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads),
            nn.Softplus()  # 항상 양수 보장
        )
    
    def forward(self, uncertainty, attention_logits):
        temp = self.temp_predictor(uncertainty) + 0.5
        temp = temp.unsqueeze(1)
        scaled_attention = attention_logits / temp
        return scaled_attention


class MultiHeadedEdgeAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_node: int, dim_edge: int, dim_atten: int, use_bn=False,
                 attention = 'fat', use_edge:bool = True, **kwargs):
        super().__init__()
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.name = 'MultiHeadedEdgeAttention'
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.d_n = d_n = dim_node // num_heads
        self.d_e = d_e = dim_edge // num_heads
        self.d_o = d_o = dim_atten // num_heads
        self.num_heads = num_heads
        self.use_edge = use_edge
        self.nn_edge = build_mlp([dim_node*2+dim_edge,(dim_node+dim_edge),dim_edge],
                          do_bn= use_bn, on_last=False)
        self.mask_obj = 0.5
        
        DROP_OUT_ATTEN = None
        if 'DROP_OUT_ATTEN' in kwargs:
            DROP_OUT_ATTEN = kwargs['DROP_OUT_ATTEN']
        
        self.attention = attention
        assert self.attention in ['fat']
        
        self.uncertainty_estimator = DualUncertaintyEstimator(dim_node)
        self.uncertainty_propagation = UncertaintyPropagation(dim_edge, num_heads)
        self.adaptive_temperature = AdaptiveTemperatureAttention(4, num_heads)
        
        if self.attention == 'fat':
            if use_edge:
                self.nn = MLP([d_n+d_e, d_n+d_e, d_o],do_bn=use_bn,drop_out = DROP_OUT_ATTEN)
            else:
                self.nn = MLP([d_n, d_n*2, d_o],do_bn=use_bn,drop_out = DROP_OUT_ATTEN)
                
            self.proj_edge  = build_mlp([dim_edge,dim_edge])
            self.proj_query = build_mlp([dim_node,dim_node])
            self.proj_value = build_mlp([dim_node,dim_atten])
            
            self.uncertainty_gate = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, num_heads),
                nn.Sigmoid()
            )
            
        elif self.attention == 'distance':
            self.proj_value = build_mlp([dim_node,dim_atten])

        
    def forward(self, query, edge, value, weight=None, istrain=False):
        batch_dim = query.size(0)
        
        query_aleatoric, query_epistemic, query_uncertainty = self.uncertainty_estimator(query)
        value_aleatoric, value_epistemic, value_uncertainty = self.uncertainty_estimator(value)
        
        edge_uncertainty = self.uncertainty_propagation(query_uncertainty, value_uncertainty)
        
        edge_feature = torch.cat([query, edge, value],dim=1)
        edge_feature = self.nn_edge(edge_feature)
        
        object_uncertainty = {
            'src_aleatoric': query_aleatoric,
            'src_epistemic': query_epistemic,
            'tgt_aleatoric': value_aleatoric,
            'tgt_epistemic': value_epistemic,
            'edge': edge_uncertainty
        }

        if self.attention == 'fat':
            value = self.proj_value(value)
            query = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads)
            edge = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)
            
            if self.use_edge:
                attn_logits = self.nn(torch.cat([query,edge],dim=1))  # b, dim, head
            else:
                attn_logits = self.nn(query)  # b, dim, head
            
            combined_uncertainty = torch.cat([query_uncertainty, value_uncertainty], dim=1)
            scaled_logits = self.adaptive_temperature(combined_uncertainty, attn_logits)
            
            prob = scaled_logits.softmax(1)
            
            confidence_weight = self.uncertainty_gate(query_uncertainty)
            confidence_weight = confidence_weight.unsqueeze(1).expand_as(prob)
            
            adjusted_prob = prob * confidence_weight
            
            x = torch.einsum('bm,bm->bm', adjusted_prob.reshape_as(value), value)
        
        elif self.attention == 'distance':
            raise NotImplementedError()
        
        else:
            raise NotImplementedError('')
        
        return x, edge_feature, prob, object_uncertainty


class MMG_single(torch.nn.Module):
    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr= 'max', 
                 use_bn=False,flow='target_to_source', attention = 'fat', 
                 hidden_size=512, depth=1, use_edge:bool=True, **kwargs):
        
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
                            **kwargs))
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
        
        self.uncertainty_aggregation = nn.Sequential(
            nn.Linear(4 * depth, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
    
    def forward(self, obj_feature_3d, edge_feature_3d, edge_index, batch_ids, obj_center=None, istrain=False):
        layer_uncertainties = []
        
        for i in range(self.depth):
            obj_feature_3d, edge_feature_3d, uncertainty = self.gcn_3ds[i](
                obj_feature_3d, edge_feature_3d, edge_index, istrain=istrain)
            
            layer_uncertainties.append(torch.cat([
                uncertainty['src_aleatoric'], 
                uncertainty['src_epistemic']
            ], dim=1))
            
            if i < (self.depth-1) or self.depth==1:
                obj_feature_3d = F.relu(obj_feature_3d)
                obj_feature_3d = self.drop_out(obj_feature_3d)
                
                edge_feature_3d = F.relu(edge_feature_3d)
                edge_feature_3d = self.drop_out(edge_feature_3d)
        
        all_uncertainties = torch.cat(layer_uncertainties, dim=1)
        final_uncertainty = self.uncertainty_aggregation(all_uncertainties)
        
        return obj_feature_3d, edge_feature_3d, final_uncertainty