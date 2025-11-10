#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genericpath import isfile
import json
import os
if __name__ == '__main__':
    os.sys.path.append('./src')
from src.model.transformer.attention import MultiHeadAttention
from src.model.model_utils.network_MMGpt import GraphEdgeAttenNetwork
from src.model.model_utils.network_MMG import GraphEdgeAttenNetwork as oGE
from src.model.model_utils.network_util import build_mlp
from src.model.model import MMGNet
from src.utils.config import Config
from utils import util
import torch
import argparse
import time
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table

def load_config():
    r"""loads model config
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config_example.json', help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('--loadbest', type=int, default=0, choices=[0,1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')
    parser.add_argument('--mode', type=str, choices=['train','trace','eval'], help='mode. can be [train,trace,eval]', required=True)
    parser.add_argument('--exp', type=str, help='experiment name')

    args = parser.parse_args()
    config_path = os.path.abspath(args.config)

    if not os.path.exists(config_path):
        raise RuntimeError('Target config file does not exist. {}'.format(config_path))
    
    #timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    config = Config(config_path)
    
    if 'NAME' not in config:
        config_name = os.path.basename(args.config)
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['NAME'] = name            
    
    config.LOADBEST = args.loadbest
    config.MODE = args.mode
    config.exp = args.exp
    '''
    if args.exp:
        config.exp = f"{timestamp}_{args.exp}"
    else:
        config.exp = timestamp '''
    
    print(f"exp name: {config.exp}")
    
    return config


def main():
    config = load_config()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    util.set_random_seed(config.SEED)

    if config.VERBOSE:
        print(config)
    
    model = MMGNet(config)
    
    ### GSE components
    self_attn = nn.ModuleList(
        MultiHeadAttention(d_model=512, d_k=512 // 8, d_v=512 // 8, h=8) 
        for i in range(2)
    )
    
    ### Graph components
    beg = GraphEdgeAttenNetwork(
        8,
        512,
        512,
        256,
        'max',
        use_bn=False,
        flow='target_to_source',
        attention="fat",
        use_edge=True,
        DROP_OUT_ATTEN=0.5
    )
    original = oGE(
        8,
        512,
        512,
        256,
        'max',
        use_bn=False,
        flow='target_to_source',
        attention="fat",
        use_edge=True,
        DROP_OUT_ATTEN=0.5
    )

    ### LSE
    proj_geo_desc = build_mlp([
        512, 
        512 // 4, 
        11
    ], do_bn=True, on_last=True)

    
    print("Our model: ", sum(p.numel() for p in model.model.parameters() if p.requires_grad))
    total_parameters = sum(p.numel() for p in model.model.parameters())
    print("total parameters: ", total_parameters)
    
    print("# of training parameters of GSE: ", sum(p.numel() for p in self_attn.parameters() if p.requires_grad))
    
    p_BEG = sum(p.numel() for p in beg.parameters() if p.requires_grad)
    p_original = sum(p.numel() for p in original.parameters() if p.requires_grad)
    print("# of training parameters of BEG: ", p_BEG - p_original)
    
    print("# of training parameters of LSE: ", sum(p.numel() for p in proj_geo_desc.parameters() if p.requires_grad))

    tmp_obj = torch.ones(2, 9, 512).float().to("cuda")
    tmp_edge = torch.ones(2, 2).long().to("cuda")
    tmp_desc = torch.ones(2, 11).float().to("cuda")
    tmp_batch_idx = torch.ones(2).long().to("cuda")
    flops = FlopCountAnalysis(model.model, (tmp_obj, tmp_edge, tmp_desc, tmp_batch_idx))

    print(flops.total()) # kb단위로 모델전체 FLOPs 출력해줌
    print(flop_count_table(flops)) # 테이블 형태로 각 연산하는 모듈마다 출력해주고, 전체도 출력해줌
    
    # summary(model.model, input_size=(9, 512))
    
if __name__ == "__main__":
    main()