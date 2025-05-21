import matplotlib.pyplot as plt
import numpy as np


def draw_graph(predicate_mean, topk_index, save_path='predicate_acc_graph.png'):
    fig, ax2 = plt.subplots(figsize=(13, 10))

    # x축: class labels
    labels = [item[0] for item in predicate_mean]
    x = np.arange(len(labels))

    # 왼쪽 y축: Accuracy
    acc_values = [item[2][topk_index]*100 for item in predicate_mean]
    ax2.set_ylabel("Accuracy (%)", color="black", fontsize=22)
    bars = ax2.bar(x, acc_values, color="C0", alpha=0.6, label="Accuracy", linewidth=1.6, edgecolor="black")
    ax2.tick_params(axis="y", labelcolor="black", labelsize=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, fontsize=24)
    
    # 전체 빈도 수 합
    total_freq = sum(item[1] for item in predicate_mean)

    # 퍼센트로 변환
    freqs_percent = [item[1] / total_freq * 100 for item in predicate_mean]

    # 연한 회색 가로선 추가
    for y in [20, 40, 60, 80]:
        ax2.axhline(y=y, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

    # 오른쪽 y축: Frequency
    ax1 = ax2.twinx()
    ax1.set_ylabel("Frequency (%)", color="#4b0055", fontsize=22)
    ax1.plot(x, freqs_percent, marker="o", linestyle="-", color="#4b0055", label="Frequency", linewidth=2.8, markersize=8)
    ax1.tick_params(axis="y", labelcolor="#4b0055", labelsize=20)

    # 전체 폰트 크기 조정 및 여백 설정
    fig.tight_layout()

    # 제목 제거
    fig.suptitle("")

    # 이미지로 저장
    fig.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig

predicate_mean = [['close by', 12448, [0.8835154241645244, 0.9915649100257069, 0.9981523136246787]], ['left', 11991, [0.9703110666333084, 0.9984988741556167, 0.9997498123592694]], ['right', 11991, [0.9701442748728213, 0.9984988741556167, 0.9998332082395129]], ['standing on', 9972, [0.9367228239069394, 0.9935820296831127, 0.998094665062174]], ['attached to', 7515, [0.9277445109780439, 0.9905522288755821, 0.9968063872255489]], ['front', 6751, [0.9004591912309288, 0.9971856021330173, 0.9997037475929492]], ['behind', 6751, [0.8963116575322174, 0.9973337283365428, 0.9998518737964746]], ['same as', 2560, [0.662890625, 0.968359375, 0.996875]], ['lying on', 2024, [0.6511857707509882, 0.9664031620553359, 0.9856719367588933]], ['higher than', 1830, [0.6901639344262295, 0.9508196721311475, 0.9950819672131147]], ['lower than', 1830, [0.6989071038251367, 0.9524590163934427, 0.9967213114754099]], ['hanging on', 1205, [0.7692946058091287, 0.966804979253112, 0.9842323651452282]], ['bigger than', 923, [0.6251354279523293, 0.942578548212351, 0.9902491874322861]], ['smaller than', 923, [0.6359696641386782, 0.9360780065005417, 0.991332611050921]], ['supported by', 822, [0.4416058394160584, 0.8844282238442822, 0.9720194647201946]], ['same symmetry as', 258, [0.6511627906976745, 0.9341085271317829, 0.9806201550387597]], ['standing in', 243, [0.6419753086419753, 0.8353909465020576, 0.9300411522633745]], ['build in', 240, [0.8791666666666667, 0.9625, 0.9666666666666667]], ['connected to', 192, [0.890625, 0.984375, 1.0]], ['leaning against', 190, [0.32105263157894737, 0.6736842105263158, 0.868421052631579]], ['belonging to', 171, [0.9005847953216374, 0.9415204678362573, 0.9649122807017544]], ['lying in', 96, [0.25, 0.6979166666666666, 0.8645833333333334]], ['part of', 66, [0.7878787878787878, 0.9393939393939394, 1.0]], ['cover', 45, [0.35555555555555557, 0.6222222222222222, 0.7333333333333333]], ['hanging in', 9, [0.3333333333333333, 0.6666666666666666, 0.7777777777777778]], ['inside', 0, [0, 0, 0]]]
draw_graph(predicate_mean, 0, save_path='acc1_graph.png')