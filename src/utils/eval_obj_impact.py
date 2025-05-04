import numpy as np
import torch
import torch.nn.functional as F

def object_entropy(obj_preds):
    """
    obj_logits : (N_obj , N_cls)   --  log_softmax 값이라고 가정
    return     : 스칼라, 장면 평균 엔트로피
    """
    obj_probs = F.softmax(obj_preds, dim=-1)
    ent       = -(obj_probs * torch.log(obj_probs + 1e-12)).sum(axis=1)
    return ent
