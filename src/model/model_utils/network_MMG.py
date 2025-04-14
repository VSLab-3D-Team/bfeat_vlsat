import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model_utils.network_util import (MLP, Aggre_Index, Gen_Index,
                                                build_mlp)
from src.model.transformer.attention import MultiHeadAttention

from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
import math
from torch import Tensor
from typing import Optional


class GraphEdgeAttenNetwork(torch.nn.Module):
    def __init__(self, num_heads, dim_node, dim_edge, dim_atten, aggr= 'max', use_bn=False,
                 flow='target_to_source',attention = 'fat',use_edge:bool=True, **kwargs):
        super().__init__() #  "Max" aggregation.
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
        xx, gcn_edge_feature, prob = self.edgeatten(x_i, edge_feature, x_j, weight, istrain=istrain)
        xx = self.index_aggr(xx, edge_index, dim_size = x.shape[0])
        xx = self.prop(torch.cat([x,xx],dim=1))
        return xx, gcn_edge_feature
  

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
            # print('drop out in',self.name,'with value',DROP_OUT_ATTEN)
        
        self.attention = attention
        assert self.attention in ['fat']
        
        if self.attention == 'fat':
            if use_edge:
                self.nn = MLP([d_n+d_e, d_n+d_e, d_o],do_bn=use_bn,drop_out = DROP_OUT_ATTEN)
            else:
                self.nn = MLP([d_n, d_n*2, d_o],do_bn=use_bn,drop_out = DROP_OUT_ATTEN)
                
            self.proj_edge  = build_mlp([dim_edge,dim_edge])
            self.proj_query = build_mlp([dim_node,dim_node])
            self.proj_value = build_mlp([dim_node,dim_atten])
        elif self.attention == 'distance':
            self.proj_value = build_mlp([dim_node,dim_atten])

        
    def forward(self, query, edge, value, weight=None, istrain=False):
        batch_dim = query.size(0)
        
        edge_feature = torch.cat([query, edge, value],dim=1)
        # avoid overfitting by mask relation input object feature
        # if random.random() < self.mask_obj and istrain: 
        #     feat_mask = torch.cat([torch.ones_like(query),torch.zeros_like(edge), torch.ones_like(value)],dim=1)
        #     edge_feature = torch.where(feat_mask == 1, edge_feature, torch.zeros_like(edge_feature))
        
        edge_feature = self.nn_edge( edge_feature )#.view(b, -1, 1)

        if self.attention == 'fat':
            value = self.proj_value(value)
            query = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads)
            edge = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)
            if self.use_edge:
                prob = self.nn(torch.cat([query,edge],dim=1)) # b, dim, head    
            else:
                prob = self.nn(query) # b, dim, head 
            prob = prob.softmax(1)
            x = torch.einsum('bm,bm->bm', prob.reshape_as(value), value)
        
        elif self.attention == 'distance':
            raise NotImplementedError()
        
        else:
            raise NotImplementedError('')
        
        return x, edge_feature, prob
    
    
class MMG(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr= 'max', 
                 use_bn=False,flow='target_to_source', attention = 'fat', 
                 hidden_size=512, depth=1, use_edge:bool=True, **kwargs,
                 ):
        
        super().__init__()

        self.num_heads = num_heads
        self.depth = depth

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads) for i in range(depth))

        self.cross_attn = nn.ModuleList(
            MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads) for i in range(depth))

        self.cross_attn_rel = nn.ModuleList(
            MultiHeadAttention(d_model=dim_edge, d_k=dim_edge // num_heads, d_v=dim_edge // num_heads, h=num_heads) for i in range(depth))
        
        self.gcn_2ds = torch.nn.ModuleList()
        self.gcn_3ds = torch.nn.ModuleList()
        
        for _ in range(self.depth):

            self.gcn_2ds.append(GraphEdgeAttenNetwork(
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
           
        self.self_attn_fc = nn.Sequential(  # 11 32 32 4(head)
            nn.Linear(4, 32),  # xyz, dist
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_heads)
        )
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
    
    
    def forward(self, obj_feature_3d, obj_feature_2d, edge_feature_3d, edge_feature_2d, edge_index, batch_ids, obj_center=None, discriptor=None, istrain=False):

        # compute weight for obj
        if obj_center is not None:
            # get attention weight for object
            batch_size = batch_ids.max().item() + 1
            N_K = obj_feature_3d.shape[0]
            obj_mask = torch.zeros(1, 1, N_K, N_K).cuda()
            obj_distance_weight = torch.zeros(1, self.num_heads, N_K, N_K).cuda()
            count = 0

            for i in range(batch_size):

                idx_i = torch.where(batch_ids == i)[0]
                obj_mask[:, :, count:count + len(idx_i), count:count + len(idx_i)] = 1
            
                center_A = obj_center[None, idx_i, :].clone().detach().repeat(len(idx_i), 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, len(idx_i), 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
                weights = torch.cat([center_dist, dist], dim=-1).unsqueeze(0)  # 1 N N 4
                dist_weights = self.self_attn_fc(weights).permute(0,3,1,2)  # 1 num_heads N N
                
                attention_matrix_way = 'add'
                obj_distance_weight[:, :, count:count + len(idx_i), count:count + len(idx_i)] = dist_weights

                count += len(idx_i)
        else:
            obj_mask = None
            obj_distance = None
            attention_matrix_way = 'mul'


        for i in range(self.depth):

            obj_feature_3d = obj_feature_3d.unsqueeze(0)
            obj_feature_2d = obj_feature_2d.unsqueeze(0)
            
            obj_feature_3d = self.self_attn[i](obj_feature_3d, obj_feature_3d, obj_feature_3d, attention_weights=obj_distance_weight, way=attention_matrix_way, attention_mask=obj_mask, use_knn=False)
            obj_feature_2d = self.cross_attn[i](obj_feature_2d, obj_feature_3d, obj_feature_3d, attention_weights=obj_distance_weight, way=attention_matrix_way, attention_mask=obj_mask, use_knn=False)
            
            obj_feature_3d = obj_feature_3d.squeeze(0)
            obj_feature_2d = obj_feature_2d.squeeze(0)  


            obj_feature_3d, edge_feature_3d = self.gcn_3ds[i](obj_feature_3d, edge_feature_3d, edge_index, istrain=istrain)
            obj_feature_2d, edge_feature_2d = self.gcn_2ds[i](obj_feature_2d, edge_feature_2d, edge_index, istrain=istrain)

            
            edge_feature_2d = edge_feature_2d.unsqueeze(0)
            edge_feature_3d = edge_feature_3d.unsqueeze(0)
            
            edge_feature_2d = self.cross_attn_rel[i](edge_feature_2d, edge_feature_3d, edge_feature_3d, use_knn=False)
            
            edge_feature_2d = edge_feature_2d.squeeze(0)
            edge_feature_3d = edge_feature_3d.squeeze(0)

            if i < (self.depth-1) or self.depth==1:
                
                obj_feature_3d = F.relu(obj_feature_3d)
                obj_feature_3d = self.drop_out(obj_feature_3d)
                
                obj_feature_2d = F.relu(obj_feature_2d)
                obj_feature_2d = self.drop_out(obj_feature_2d)

                edge_feature_3d = F.relu(edge_feature_3d)
                edge_feature_3d = self.drop_out(edge_feature_3d)

                edge_feature_2d = F.relu(edge_feature_2d)
                edge_feature_2d = self.drop_out(edge_feature_2d)
        
        return obj_feature_3d, obj_feature_2d, edge_feature_3d, edge_feature_2d

class BidirectionalEdgeLayer(MessagePassing):
    def __init__(self,
                 dim_node: int, dim_edge: int, dim_atten: int,
                 num_heads: int,
                 use_bn: bool = True,
                 aggr='max',
                 attn_dropout: float = 0.3,
                 flow: str = 'target_to_source',
                 use_distance_mask: bool = True,
                 use_node_attention: bool = False):
        super().__init__(aggr=aggr, flow=flow)
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.dim_node_proj = dim_node // num_heads
        self.dim_edge_proj = dim_edge // num_heads
        self.dim_value_proj = dim_atten // num_heads
        self.num_head = num_heads
        self.temperature = math.sqrt(self.dim_edge_proj)
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_atten = dim_atten
        self.use_distance_mask = use_distance_mask
        self.use_node_attention = use_node_attention

        self.proj_q = build_mlp([dim_node, dim_node])
        self.proj_v = build_mlp([dim_node, dim_atten])
        self.proj_k = build_mlp([dim_edge, dim_edge])
        
        if self.use_distance_mask:
            self.distance_mlp = build_mlp([4, 32, 1], do_bn=use_bn, on_last=False)
        
        if self.use_node_attention:
            self.node_distance_mlp = build_mlp([1, 32, 1], do_bn=False, on_last=False)
            
            self.mhsa_q = torch.nn.Linear(dim_node, dim_node)
            self.mhsa_k = torch.nn.Linear(dim_node, dim_node)
            self.mhsa_v = torch.nn.Linear(dim_node, dim_node)
            
            self.mhsa_out = torch.nn.Linear(dim_node, dim_node)
            
            self.layer_norm1 = torch.nn.LayerNorm(dim_node)
            self.layer_norm2 = torch.nn.LayerNorm(dim_node)
            
            self.ffn = torch.nn.Sequential(
                torch.nn.Linear(dim_node, dim_node * 4),
                torch.nn.ReLU(),
                torch.nn.Dropout(attn_dropout),
                torch.nn.Linear(dim_node * 4, dim_node)
            )
        
        self.nn_edge_update = build_mlp([dim_node*2+dim_edge*2, dim_node+dim_edge*2, dim_edge],
                                       do_bn=use_bn, on_last=False)
        
        self.edge_attention_mlp = build_mlp([dim_edge*2, dim_edge], do_bn=use_bn, on_last=False)
        
        self.nn_node_update = build_mlp([dim_node+dim_edge, dim_node+dim_edge, dim_node],
                                       do_bn=use_bn, on_last=False)
        
        self.nn_att = MLP([self.dim_node_proj+self.dim_edge_proj, 
                          self.dim_node_proj+self.dim_edge_proj,
                          self.dim_edge_proj])
        
        self.dropout = torch.nn.Dropout(
            attn_dropout) if attn_dropout > 0 else torch.nn.Identity()
        
        self.sigmoid = torch.nn.Sigmoid()

    def create_node_distance_mask(self, node_positions):
        if not self.use_node_attention:
            return None
            
        num_nodes = node_positions.size(0)
        distance_matrix = torch.zeros((num_nodes, num_nodes), device=node_positions.device)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                diff = node_positions[i] - node_positions[j]
                distance_matrix[i, j] = torch.norm(diff, p=2)
        
        input_tensor = distance_matrix.view(-1, 1)
        output = self.node_distance_mlp(input_tensor)
        
        attention_mask = self.sigmoid(output).view(num_nodes, num_nodes)
        return attention_mask
    
    def apply_mhsa_with_distance_mask(self, x, distance_mask=None):
        if not self.use_node_attention:
            return x
            
        batch_size, seq_len = x.size(0), x.size(0)  # batch_size = 노드 수, seq_len = 노드 수
        head_dim = self.dim_node // self.num_head
        
        residual = x
        x = self.layer_norm1(x)
        
        q = self.mhsa_q(x).view(batch_size, self.num_head, head_dim)  # [N, H, D/H]
        k = self.mhsa_k(x).view(batch_size, self.num_head, head_dim)  # [N, H, D/H]
        v = self.mhsa_v(x).view(batch_size, self.num_head, head_dim)  # [N, H, D/H]
        
        q = q.permute(1, 0, 2)  # [H, N, D/H]
        k = k.permute(1, 0, 2)  # [H, N, D/H]
        v = v.permute(1, 0, 2)  # [H, N, D/H]
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [H, N, N]
        
        if distance_mask is not None:
            distance_mask = distance_mask.unsqueeze(0).expand(self.num_head, -1, -1)  # [H, N, N]
            attn_scores = attn_scores * distance_mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [H, N, N]
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)  # [H, N, D/H]
        
        out = out.permute(1, 0, 2).contiguous().view(batch_size, -1)  # [N, D]
        
        out = self.mhsa_out(out)
        out = self.dropout(out)
        
        out = out + residual
        
        residual = out
        out = self.layer_norm2(out)
        out = self.ffn(out)
        out = self.dropout(out)
        out = out + residual
        
        return out

    def forward(self, x, edge_feature, edge_index, node_positions=None):
        row, col = edge_index
        
        edge_id_mapping = {}
        for idx, (i, j) in enumerate(zip(row, col)):
            edge_id_mapping[(i.item(), j.item())] = idx
        
        reverse_edge_feature = torch.zeros_like(edge_feature)
        
        for idx, (i, j) in enumerate(zip(row, col)):
            if (j.item(), i.item()) in edge_id_mapping:
                reverse_idx = edge_id_mapping[(j.item(), i.item())]
                reverse_edge_feature[idx] = edge_feature[reverse_idx]
        
        distance_mask = None
        if self.use_distance_mask and node_positions is not None:
            distance_features = []
            for i, j in zip(row, col):
                i, j = i.item(), j.item()
                pos_i, pos_j = node_positions[i], node_positions[j]
                diff = pos_i - pos_j
                dist = torch.norm(diff, p=2)
                distance_features.append(torch.cat([diff, dist.unsqueeze(0)], dim=0))
            
            distance_features = torch.stack(distance_features)
            
            distance_mask = self.distance_mlp(distance_features)
            distance_mask = self.sigmoid(distance_mask).squeeze(-1)
        
        outgoing_edges = {}
        incoming_edges = {}
        
        for idx, (i, j) in enumerate(zip(row, col)):
            i, j = i.item(), j.item()
            if i not in outgoing_edges:
                outgoing_edges[i] = []
            outgoing_edges[i].append((idx, j))
            
            if j not in incoming_edges:
                incoming_edges[j] = []
            incoming_edges[j].append((idx, i))
        
        updated_node, updated_edge, prob = self.propagate(
            edge_index, 
            x=x, 
            edge_feature=edge_feature,
            reverse_edge_feature=reverse_edge_feature,
            distance_mask=distance_mask,
            x_ori=x
        )
        
        if self.use_node_attention and node_positions is not None:
            node_distance_mask = self.create_node_distance_mask(node_positions)
            updated_node = self.apply_mhsa_with_distance_mask(updated_node, node_distance_mask)
        
        twin_edge_attention = torch.zeros((x.size(0), self.dim_edge*2), device=x.device)
        
        for node_id in range(x.size(0)):
            outgoing_feature = torch.zeros(self.dim_edge, device=x.device)
            if node_id in outgoing_edges:
                for edge_idx, _ in outgoing_edges[node_id]:
                    outgoing_feature += updated_edge[edge_idx]
                if len(outgoing_edges[node_id]) > 0:
                    outgoing_feature /= len(outgoing_edges[node_id])
            
            incoming_feature = torch.zeros(self.dim_edge, device=x.device)
            if node_id in incoming_edges:
                for edge_idx, _ in incoming_edges[node_id]:
                    incoming_feature += updated_edge[edge_idx]
                if len(incoming_edges[node_id]) > 0:
                    incoming_feature /= len(incoming_edges[node_id])
            
            twin_edge_attention[node_id] = torch.cat([outgoing_feature, incoming_feature], dim=0)
        
        edge_attention = self.edge_attention_mlp(twin_edge_attention)
        edge_attention = self.sigmoid(edge_attention)
        
        node_feature_nonlinear = torch.nn.functional.relu(updated_node)  # f(v_i^l)
        final_node = node_feature_nonlinear * edge_attention  # ⊙ β(A_ε)
        
        return final_node, updated_edge, prob

    def message(self, x_i: Tensor, x_j: Tensor, 
                edge_feature: Tensor, reverse_edge_feature: Tensor,
                distance_mask: Optional[Tensor] = None) -> Tensor:
        '''
        x_i: 소스 노드 특징 [N, D_N]
        x_j: 타겟 노드 특징 [N, D_N]
        edge_feature: 정방향 에지 특징 [N, D_E]
        reverse_edge_feature: 역방향 에지 특징 [N, D_E]
        distance_mask: 거리 기반 마스킹 가중치 [N] (선택적)
        '''
        num_edge = x_i.size(0)
        
        updated_edge = self.nn_edge_update(
            torch.cat([x_i, edge_feature, reverse_edge_feature, x_j], dim=1)
        )
        
        x_i_proj = self.proj_q(x_i).view(
            num_edge, self.dim_node_proj, self.num_head)  # [N, D, H]
        edge_proj = self.proj_k(edge_feature).view(
            num_edge, self.dim_edge_proj, self.num_head)  # [N, D, H]
        x_j_val = self.proj_v(x_j)
        
        att = self.nn_att(torch.cat([x_i_proj, edge_proj], dim=1))  # [N, D, H]
        
        if distance_mask is not None:
            distance_mask = distance_mask.view(-1, 1, 1)
            att = att * distance_mask
        
        prob = torch.nn.functional.softmax(att/self.temperature, dim=1)
        prob = self.dropout(prob)
        
        weighted_value = prob.reshape_as(x_j_val) * x_j_val
        
        return [weighted_value, updated_edge, prob]

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        weighted_value, updated_edge, prob = inputs
        weighted_value = scatter(weighted_value, index, dim=self.node_dim,
                                dim_size=dim_size, reduce=self.aggr)
        return weighted_value, updated_edge, prob

    def update(self, inputs, x_ori):
        weighted_value, updated_edge, prob = inputs
        
        updated_node = self.nn_node_update(
            torch.cat([x_ori, weighted_value], dim=1)
        )
        
        return updated_node, updated_edge, prob

class MMG_single(torch.nn.Module):
    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=8, aggr='max', 
                 use_bn=True, flow='target_to_source', attention='fat', 
                 hidden_size=512, depth=2, use_edge=True, **kwargs):
        
        super().__init__()

        self.num_heads = num_heads
        self.depth = depth
        self.use_distance_mask = kwargs.get('use_distance_mask', True)
        self.use_node_attention = kwargs.get('use_node_attention', False)
        
        self.gcn_3ds = torch.nn.ModuleList()
        
        for _ in range(self.depth):
            self.gcn_3ds.append(BidirectionalEdgeLayer(
                dim_node=dim_node,
                dim_edge=dim_edge,
                dim_atten=dim_atten,
                num_heads=num_heads,
                use_bn=use_bn,
                aggr=aggr,
                attn_dropout=kwargs.get('DROP_OUT_ATTEN', 0.1),
                flow=flow,
                use_distance_mask=self.use_distance_mask,
                use_node_attention=self.use_node_attention
            ))
        
        self.drop_out = torch.nn.Dropout(kwargs.get('DROP_OUT_ATTEN', 0.1))
    
    def forward(self, obj_feature_3d, edge_feature_3d, edge_index, batch_ids, obj_center=None, istrain=False):
        
        node_positions = None
        if obj_center is not None and (self.use_distance_mask or self.use_node_attention):
            node_positions = obj_center
        
        for i in range(self.depth):
            obj_feature_3d, edge_feature_3d, _ = self.gcn_3ds[i](
                obj_feature_3d, edge_feature_3d, edge_index, node_positions
            )
            
            if i < (self.depth-1) or self.depth==1:
                obj_feature_3d = F.relu(obj_feature_3d)
                obj_feature_3d = self.drop_out(obj_feature_3d)
                
                edge_feature_3d = F.relu(edge_feature_3d)
                edge_feature_3d = self.drop_out(edge_feature_3d)
        
        return obj_feature_3d, edge_feature_3d

 
class MMG_teacher(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr= 'max', 
                 use_bn=False,flow='target_to_source', attention = 'fat', 
                 hidden_size=512, depth=1, use_edge:bool=True, **kwargs,
                 ):
        
        super().__init__()

        self.num_heads = num_heads
        self.depth = depth

        self.self_attn_3d = MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads)

        self.cross_attn_3d = MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads)
        
        self.self_attn_2d = MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads)

        self.cross_attn_2d = MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads)

        self.fusion_module = nn.Sequential(
            nn.Linear(512 * 4, 512 * 2),
            nn.ReLU(),
            nn.BatchNorm1d(512 * 2),
            nn.Dropout(0.5),
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )

        self.gcns = torch.nn.ModuleList()   
        for _ in range(self.depth):
            
            self.gcns.append(GraphEdgeAttenNetwork(
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
           
        self.self_attn_fc = nn.Sequential(  # 4 32 32 4(head)
            nn.Linear(4, 32),  # xyz, dist
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_heads)
        )
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
    
    def forward(self, obj_feature_3d, obj_feature_2d, edge_feature, edge_index, batch_ids, obj_center=None, istrain=False):

        if obj_center is not None:
            # get attention weight
            batch_size = batch_ids.max().item() + 1
            N_K = obj_feature_3d.shape[0]
            mask = torch.zeros(1, 1, N_K, N_K).cuda()
            distance = torch.zeros(1, self.num_heads, N_K, N_K).cuda()
            count = 0

            for i in range(batch_size):

                idx_i = torch.where(batch_ids == i)[0]
                mask[:, :, count:count + len(idx_i), count:count + len(idx_i)] = 1
            
                center_A = obj_center[None, idx_i, :].clone().detach().repeat(len(idx_i), 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, len(idx_i), 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
                weights = torch.cat([center_dist, dist], dim=-1).unsqueeze(0)  # 1 N N 4
                dist_weights = self.self_attn_fc(weights).permute(0,3,1,2)  # 1 num_heads N N
                
                attention_matrix_way = 'add'
                distance[:, :, count:count + len(idx_i), count:count + len(idx_i)] = dist_weights

                count += len(idx_i)
        else:
            mask = None
            distance = None
            attention_matrix_way = 'mul'
         
        obj_feature_3d = obj_feature_3d.unsqueeze(0)
        obj_feature_2d = obj_feature_2d.unsqueeze(0)
        
        obj_feature_3d_sa = self.self_attn_3d(obj_feature_3d, obj_feature_3d, obj_feature_3d, attention_weights=distance, way=attention_matrix_way, attention_mask=mask)
        obj_feature_2d_sa = self.self_attn_2d(obj_feature_2d, obj_feature_2d, obj_feature_2d, attention_weights=distance, way=attention_matrix_way, attention_mask=mask)
        obj_feature_3d_ca = self.cross_attn_3d(obj_feature_3d_sa, obj_feature_2d_sa, obj_feature_2d_sa, attention_weights=distance, way=attention_matrix_way, attention_mask=mask)
        obj_feature_2d_ca = self.cross_attn_2d(obj_feature_2d_sa, obj_feature_3d_sa, obj_feature_3d_sa, attention_weights=distance, way=attention_matrix_way, attention_mask=mask)
        
        obj_feature_3d_sa = obj_feature_3d_sa.squeeze(0)
        obj_feature_2d_sa = obj_feature_2d_sa.squeeze(0)
        obj_feature_3d_ca = obj_feature_3d_ca.squeeze(0)
        obj_feature_2d_ca = obj_feature_2d_ca.squeeze(0)
        
        # fusion 3d and 2d
        obj_feature = self.fusion_module(torch.cat([obj_feature_3d_sa, obj_feature_2d_sa, obj_feature_3d_ca, obj_feature_2d_ca], dim=-1))
        obj_feature_mimic = obj_feature.clone().detach()
        
        for i in range(self.depth):

            obj_feature, edge_feature = self.gcns[i](obj_feature, edge_feature, edge_index, istrain=istrain)

            if i < (self.depth-1) or self.depth==1:
                
                obj_feature = F.relu(obj_feature)
                obj_feature = self.drop_out(obj_feature)
                
                edge_feature = F.relu(edge_feature)
                edge_feature = self.drop_out(edge_feature)
        
        return obj_feature, edge_feature, obj_feature_mimic


class MMG_student(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr= 'max', 
                 use_bn=False,flow='target_to_source', attention = 'fat', 
                 hidden_size=512, depth=1, use_edge:bool=True, **kwargs,
                 ):
        
        super().__init__()

        self.num_heads = num_heads
        self.depth = depth

        self.self_attn_before = MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads)
        self.self_attn_after = MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads)
        self.gcns = torch.nn.ModuleList()
        
        for _ in range(self.depth):
            
            self.gcns.append(GraphEdgeAttenNetwork(
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
           
        self.self_attn_fc = nn.Sequential(  # 4 32 32 4(head)
            nn.Linear(4, 32),  # xyz, dist
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_heads)
        )

        self.modality_learner = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512)
        )

        self.mlp = nn.Linear(512 * 2, 512)
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
    
    def forward(self, obj_feature, edge_feature, edge_index, batch_ids, obj_center=None, istrain=False):
        
        if obj_center is not None:
            # get attention weight
            batch_size = batch_ids.max().item() + 1
            N_K = obj_feature.shape[0]
            mask = torch.zeros(1, 1, N_K, N_K).cuda()
            distance = torch.zeros(1, self.num_heads, N_K, N_K).cuda()
            count = 0

            for i in range(batch_size):

                idx_i = torch.where(batch_ids == i)[0]
                mask[:, :, count:count + len(idx_i), count:count + len(idx_i)] = 1
            
                center_A = obj_center[None, idx_i, :].clone().detach().repeat(len(idx_i), 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, len(idx_i), 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
                weights = torch.cat([center_dist, dist], dim=-1).unsqueeze(0)  # 1 N N 4
                dist_weights = self.self_attn_fc(weights).permute(0,3,1,2)  # 1 num_heads N N
                
                attention_matrix_way = 'add'
                distance[:, :, count:count + len(idx_i), count:count + len(idx_i)] = dist_weights

                count += len(idx_i)
        else:
            mask = None
            distance = None
            attention_matrix_way = 'mul'
 
        
        obj_feature = obj_feature.unsqueeze(0)
        obj_feature = self.self_attn_before(obj_feature, obj_feature, obj_feature, attention_weights=distance, way=attention_matrix_way, attention_mask=mask)
        obj_feature = obj_feature.squeeze(0)

        #obj_feature_tmp = obj_feature.clone()
        #obj_feature_mimic = self.modality_learner(obj_feature)
        obj_feature_mimic = obj_feature.clone()
        # obj_feature_tmp = torch.cat([obj_feature_tmp, obj_feature_mimic.detach()], dim=-1)
        # obj_feature = self.mlp(obj_feature_tmp)
        
        obj_feature = obj_feature.unsqueeze(0)
        obj_feature = self.self_attn_after(obj_feature, obj_feature, obj_feature, attention_weights=distance, way=attention_matrix_way, attention_mask=mask)
        obj_feature = obj_feature.squeeze(0)
        
        for i in range(self.depth):

            obj_feature, edge_feature = self.gcns[i](obj_feature, edge_feature, edge_index, istrain=istrain)

            if i < (self.depth-1) or self.depth==1:
                
                obj_feature = F.relu(obj_feature)
                obj_feature = self.drop_out(obj_feature)
                
                edge_feature = F.relu(edge_feature)
                edge_feature = self.drop_out(edge_feature)
        
        return obj_feature, edge_feature, obj_feature_mimic
