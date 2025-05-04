import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model.model_utils.model_base import BaseModel
from utils import op_utils
from src.utils.eva_utils_acc import get_gt, evaluate_topk_object, evaluate_topk_predicate, evaluate_triplet_topk
from src.model.model_utils.network_TripletGCN import TripletGCNModel
from src.model.model_utils.network_PointNet import PointNetfeat, PointNetCls, PointNetRelCls, PointNetRelClsMulti
from src.utils.eva_utils_acc import (evaluate_topk_object,
                                 evaluate_topk_predicate,
                                 evaluate_triplet_topk, get_gt)
from src.utils.eval_utils_recall import *
from src.utils.eval_obj_impact import *

class SGPN(BaseModel):
    """
    512 + 256 baseline
    """
    def __init__(self, config, num_obj_class, num_rel_class, dim_descriptor=11):
        super().__init__('SGPN', config)

        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN

        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        
        dim_point = 3
        dim_point_rel = 3
        if mconfig.USE_RGB:
            dim_point +=3
            dim_point_rel+=3
        if mconfig.USE_NORMAL:
            dim_point +=3
            dim_point_rel+=3
            
        if mconfig.USE_CONTEXT:
            dim_point_rel += 1

        dim_point_feature = 512
        
        # Object Encoder
        self.obj_encoder = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)      
        
        # Relationship Encoder
        self.rel_encoder = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.edge_feature_size)
        
        self.gcn = TripletGCNModel(5, dim_node=512, dim_edge=256, dim_hidden=256)

        self.obj_predictor = PointNetCls(num_obj_class, in_size=dim_point_feature,
                                 batch_norm=with_bn, drop_out=True)

        if mconfig.multi_rel_outputs:
            self.rel_predictor = PointNetRelClsMulti(
                num_rel_class, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
        else:
            self.rel_predictor = PointNetRelCls(
                num_rel_class, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
        
        self.optimizer = optim.AdamW([
            {'params':self.obj_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.gcn.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            #{'params':self.mlp.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()


    def forward(self, obj_points, rel_points, edge_indices):

        obj_feature = self.obj_encoder(obj_points)
        
        rel_feature = self.rel_encoder(rel_points)

        gcn_obj_feature, gcn_rel_feature = self.gcn(obj_feature, rel_feature, edge_indices)
        
        rel_cls = self.rel_predictor(gcn_rel_feature)

        obj_logits = self.obj_predictor(obj_feature)

        return obj_logits, rel_cls

    def process_train(self, obj_points, obj_2d_feats, gt_cls, rel_points, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration +=1    
        
        obj_pred, rel_pred = self(obj_points, rel_points, edge_indices.t().contiguous())
        
        # compute loss for obj
        loss_obj = F.nll_loss(obj_pred, gt_cls)

         # compute loss for rel
        if self.mconfig.multi_rel_outputs:
            loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls)
        else:
            loss_rel = F.nll_loss(rel_pred, gt_rel_cls)

        
        loss = 0.1 * loss_obj + loss_rel
        self.backward(loss)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        

        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        
        
        log = [("train/rel_loss", loss_rel.detach().item()),
                ("train/obj_loss", loss_obj.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
            ]
        return log
           
    def process_val(self, obj_points, obj_2d_feats, gt_cls, rel_points, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, use_triplet=False):
 
        obj_pred, rel_pred = self(obj_points, rel_points, edge_indices.t().contiguous())
        
        obj_cls_viz=[]
        rel_cls_viz=[]
        
        objs_pred=obj_pred.detach().cpu()
        rels_preds=rel_pred.detach().cpu()

        topk = 10
        for obj in range(len(objs_pred)):
            res = []
            _obj_pred = objs_pred[obj]
            sorted_idx = torch.sort(_obj_pred, descending=True)[1]
            for idx in sorted_idx[:topk]:
                res.append(idx)
            obj_cls_viz.append(res)
        
        for rel in range(len(rels_preds)):
            res = []
            _rel_pred = rels_preds[rel]
            _, sorted_idx = torch.sort(_rel_pred, descending=True)
            
            for idx in sorted_idx[:topk]:
                if _rel_pred[idx]>0.5:
                    res.append(idx)
                else:
                    res.append(-1)
            rel_cls_viz.append(res)
        
        entropy_obj_scene = object_entropy(obj_pred.detach()).cpu().numpy()
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach().cpu(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        
        sgcls_recall_w = evaluate_triplet_recallk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20,50,100], 1, use_clip=True, evaluate='triplet')
        predcls_recall_w = evaluate_triplet_recallk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20,50,100], 1, use_clip=True, evaluate='rels')
        sgcls_recall_wo = evaluate_triplet_recallk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20,50,100], 1000, use_clip=True, evaluate='triplet')
        predcls_recall_wo = evaluate_triplet_recallk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20,50,100], 1000, use_clip=True, evaluate='rels')
        sgcls_mean_recall_w = evaluate_triplet_mrecallk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20,50,100], 1, use_clip=True, evaluate='triplet')
        predcls_mean_recall_w = evaluate_triplet_mrecallk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20,50,100], 1, use_clip=True, evaluate='rels')
        sgcls_mean_recall_wo = evaluate_triplet_mrecallk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20,50,100], 1000, use_clip=True, evaluate='triplet')
        predcls_mean_recall_wo = evaluate_triplet_mrecallk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20,50,100], 1000, use_clip=True, evaluate='rels')
        
        if use_triplet:
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=False, obj_topk=top_k_obj)
        else:
            top_k_triplet = [101]
            cls_matrix = None
            sub_scores = None
            obj_scores = None
            rel_scores = None

        return top_k_obj, top_k_obj, \
            top_k_rel, top_k_rel, \
            top_k_triplet, top_k_triplet, \
            cls_matrix, sub_scores, obj_scores, rel_scores, \
            sgcls_recall_w, predcls_recall_w, sgcls_recall_wo, predcls_recall_wo, \
            sgcls_mean_recall_w, predcls_mean_recall_w, sgcls_mean_recall_wo, predcls_mean_recall_wo, \
            obj_cls_viz, rel_cls_viz, entropy_obj_scene
    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()
