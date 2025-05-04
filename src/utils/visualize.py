import os
import numpy as np

def save_gt(i, gt_class, gt_rel_cls, edge_indices, visualization_path="/data/spoiuy3/vlsat/viz_sgpn"):
    save_gt_path=os.path.join(visualization_path,str(i))
    os.makedirs(save_gt_path,exist_ok=True)
    
    np.save(os.path.join(save_gt_path,'gt_class.npy'), gt_class.detach().cpu().numpy())
    np.save(os.path.join(save_gt_path,'gt_rel_cls.npy'), gt_rel_cls.detach().cpu().numpy())
    np.save(os.path.join(save_gt_path,'edge_indices.npy'), edge_indices.detach().cpu().numpy())
    
def save_prediction(i, cls, rel_cls, obj_entropy, visualization_path="/data/spoiuy3/vlsat/viz_sgpn"):
    save_gt_path=os.path.join(visualization_path,str(i))
    os.makedirs(save_gt_path,exist_ok=True)
    
    np.save(os.path.join(save_gt_path,'class.npy'), cls)
    np.save(os.path.join(save_gt_path,'rel_cls.npy'), rel_cls)
    np.save(os.path.join(save_gt_path,'entropy.npy'), obj_entropy)
    
def save_scan(i, loglist, gt_class, gt_rel_cls, edge_indices, cls, rel_cls, obj_entropy, top_k_obj, top_k_obj_2d, top_k_rel, top_k_rel_2d, tok_k_triplet, top_k_2d_triplet):
    logs_per_scan = [("Acc@1/obj_cls_acc", (top_k_obj <= 1).sum() * 100 / len(top_k_obj)),
            ("Acc@1/obj_cls_2d_acc", (top_k_obj_2d <= 1).sum() * 100 / len(top_k_obj_2d)),
            ("Acc@5/obj_cls_acc", (top_k_obj <= 5).sum() * 100 / len(top_k_obj)),
            ("Acc@5/obj_cls_2d_acc", (top_k_obj_2d <= 5).sum() * 100 / len(top_k_obj_2d)),
            ("Acc@10/obj_cls_acc", (top_k_obj <= 10).sum() * 100 / len(top_k_obj)),
            ("Acc@10/obj_cls_2d_acc", (top_k_obj_2d <= 10).sum() * 100 / len(top_k_obj_2d)),
            ("Acc@1/rel_cls_acc", (top_k_rel <= 1).sum() * 100 / len(top_k_rel)),
            ("Acc@1/rel_cls_2d_acc", (top_k_rel_2d <= 1).sum() * 100 / len(top_k_rel_2d)),
            ("Acc@3/rel_cls_acc", (top_k_rel <= 3).sum() * 100 / len(top_k_rel)),
            ("Acc@3/rel_cls_2d_acc", (top_k_rel_2d <= 3).sum() * 100 / len(top_k_rel_2d)),
            ("Acc@5/rel_cls_acc", (top_k_rel <= 5).sum() * 100 / len(top_k_rel)),
            ("Acc@5/rel_cls_2d_acc", (top_k_rel_2d <= 5).sum() * 100 / len(top_k_rel_2d)),
            ("Acc@50/triplet_acc", (tok_k_triplet <= 50).sum() * 100 / len(tok_k_triplet)),
            ("Acc@50/triplet_2d_acc", (top_k_2d_triplet <= 50).sum() * 100 / len(top_k_2d_triplet)),
            ("Acc@100/triplet_acc", (tok_k_triplet <= 100).sum() * 100 / len(tok_k_triplet)),
            ("Acc@100/triplet_2d_acc", (top_k_2d_triplet <= 100).sum() * 100 / len(top_k_2d_triplet)),]
    
    tmp=[]
    for _,j in logs_per_scan:
        tmp.append(j)
    loglist.append(tmp)
    save_gt(i, gt_class, gt_rel_cls, edge_indices)
    save_prediction(i, cls, rel_cls, obj_entropy)
    
def save_log(loglist, visualization_path="/data/spoiuy3/vlsat/viz_sgpn"):
    np.save(os.path.join(visualization_path,'logs.npy'), loglist)