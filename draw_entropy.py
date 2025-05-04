import numpy as np
import torch
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_path = "/data/spoiuy3/vlsat/viz_sgpn"
def Predicate_Object_Correlation(topk=1):
    wrong_count=[0, 0, 0]
    total_count=[0, 0, 0]
    
    entropy_pred_graph = []
    for i in range(548):
        save_gt_path=os.path.join(data_path, str(i))

        gt_class = np.load(os.path.join(save_gt_path,"gt_class.npy"))
        classses = np.load(os.path.join(save_gt_path,"class.npy"))
        edge_indices = np.load(os.path.join(save_gt_path,"edge_indices.npy"))
        rel_cls = np.load(os.path.join(save_gt_path,"rel_cls.npy"))
        gt_rel_cls = np.load(os.path.join(save_gt_path,"gt_rel_cls.npy"))
        entropy_obj = np.load(os.path.join(save_gt_path, "entropy.npy"))
        
        for i, edge in enumerate(edge_indices):
            true_flag=False
            for v in rel_cls[i]:
                if gt_rel_cls[i][v]==1 and v!=-1:
                    true_flag=True
            if 1 not in gt_rel_cls[i] and rel_cls[i][0]==-1:
                true_flag=True
            
            sub_idx, obj_idx = edge[0], edge[1]
            sub_ent, obj_ent = entropy_obj[sub_idx], entropy_obj[obj_idx]
            if not true_flag:
                if gt_class[sub_idx]in classses[sub_idx][:topk] and gt_class[obj_idx]in classses[obj_idx][:topk]:
                    wrong_count[0] += 1
                    entropy_pred_graph.append(((sub_ent + obj_ent) / 2, 1))
                elif gt_class[sub_idx] not in classses[sub_idx][:topk] and gt_class[obj_idx] not in classses[obj_idx][:topk]:
                    wrong_count[1] += 1
                else:
                    wrong_count[2] += 1
            else: 
                if gt_class[sub_idx]in classses[sub_idx][:topk] and gt_class[obj_idx]in classses[obj_idx][:topk]:
                    entropy_pred_graph.append(((sub_ent + obj_ent) / 2, 0))
                
            if gt_class[sub_idx]in classses[sub_idx][:topk] and gt_class[obj_idx]in classses[obj_idx][:topk]:
                total_count[0] += 1
            elif gt_class[sub_idx] not in classses[sub_idx][:topk] and gt_class[obj_idx] not in classses[obj_idx][:topk]:
                total_count[1] += 1
            else:
                total_count[2] += 1
            
            
    print(f"topk : {topk}") #   
    print(f"2 correct node : {wrong_count[0]/total_count[0]:0.2f} / 1 correct node : {wrong_count[2]/total_count[2]:0.2f} / 0 correct node : {wrong_count[1]/total_count[1]:0.2f}")
    return entropy_pred_graph

if __name__ == "__main__":
    entropy_pred_graph = Predicate_Object_Correlation()
    # --------------------------------------------------------------
    # 1) 입력 준비  (예시)
    # --------------------------------------------------------------
    ent, wrong = zip(*entropy_pred_graph)
    ent   = np.array(ent)
    wrong = np.array(wrong)      # 1=틀림, 0=정답

    # --------------------------------------------------------------
    # 2) bin 경계 정의
    #    - 자동: bins='auto' 또는 정수
    #    - 수동: np.linspace(min,max,N+1)
    # --------------------------------------------------------------
    num_bins = 30
    bin_edges = np.linspace(ent.min(), ent.max(), num_bins + 1)  # 등간격

    # --------------------------------------------------------------
    # 3) 빈도(hist)와 오류율 계산
    # --------------------------------------------------------------
    #   hist       : 각 bin 빈도
    #   bin_idx    : 샘플이 속한 bin 인덱스 (0 .. num_bins-1)
    hist, _      = np.histogram(ent, bins=bin_edges)
    bin_idx      = np.digitize(ent, bins=bin_edges, right=False) - 1

    # 오류율 = (bin별 wrong 합) / 빈도
    error_counts = np.zeros(num_bins, dtype=int)
    for i in range(num_bins):
        error_counts[i] = wrong[bin_idx == i].sum()

    error_rate = error_counts / np.maximum(hist, 1)  # division by zero 방지
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # --------------------------------------------------------------
    # 4) 시각화
    # --------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(7, 4))

    # 히스토그램 (빈도)
    ax1.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.6,
            label="Sample count", color="C0", edgecolor="black")
    ax1.set_xlabel("Entropy")
    ax1.set_ylabel("Count", color="C0")
    ax1.tick_params(axis='y', labelcolor="C0")

    # 오류율 (line, 0~1)
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, error_rate, marker='o', linestyle='-', color="C1",
            label="Error rate")
    ax2.set_ylabel("Error rate", color="C1")
    ax2.tick_params(axis='y', labelcolor="C1")
    ax2.set_ylim(0, 0.5)

    # 범례 합치기
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Entropy histogram and per-bin error rate")
    plt.tight_layout()
    plt.savefig("./hypothesis.png")
