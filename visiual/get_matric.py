import json
import numpy as np
from matplotlib import pyplot as plt
from get_op_list import get_op_seq
import heapq


def visual(data, classes='0123456789abcdefghijklmnopqrstuvwxyz'):
    classes = list(classes)  # + ['-']
    hot = np.zeros((len(classes), len(classes)))
    # token_confs = np.zeros((len(classes), len(classes)))
    for d in data:
        pred, gt = d[2], d[3].lower()
        # token_conf = d[1]
        if gt == pred:
            continue
        op_str = get_op_seq(pred, gt)
        # gt, pred, token_conf, op_str = list(map(list, [gt, pred, token_conf, op_str]))
        gt, pred, op_str = list(map(list, [gt, pred, op_str]))
        for op in op_str:
            if op == 's':
                gg = gt.pop(0)
                pp = pred.pop(0)
                # token_conf = token_conf.pop(0)
                hot[classes.index(gg)][classes.index(pp)] += 1
                # token_confs[classes.index(gg)][classes.index(pp)] += token_conf
            elif op == '#':
                gg = gt.pop(0)
                pp = pred.pop(0)
                # hot[classes.index(gg)][classes.index(pp)] += 1
            elif op == 'd':
                gg = '-'
                pp = pred.pop(0)
                # hot[classes.index(gg)][classes.index(pp)] += 1
            else:
                gg = gt.pop(0)
                pp = '-'
                # hot[classes.index(gg)][classes.index(pp)] += 1

    for i in range(len(classes)):
        hot[i, :] = hot[i, :] / (hot[i, :].sum() + 1e-16)

    return hot


def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues, path='confuse_matric.jpg'):
    plt.rc('font', family='Times New Roman', size='8')  # 设置字体样式、大小

    # 按行进行归一化
    cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 10e-6)
    # print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    str_cm_copy = str_cm
    cm_copy = cm.copy()
    # for row in str_cm:
    #     print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         if int(cm[i, j]*100 + 0.5) == 0:
    #             cm[i, j]=0
    '''for i in range(10):
        gt_idx, pred_idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
        print(classes[gt_idx], classes[pred_idx], np.max(cm_copy))
        cm_copy[gt_idx, pred_idx] = 0'''

    '''
    for i in range(len(str_cm)):
        for j in range(10):
            pred_max = max(str_cm[i])
            pred_max_id = str_cm[i].index(pred_max)
        print(classes[i], classes[pred_max_id])
    '''
    fig, ax = plt.subplots()
    # im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    im = ax.imshow(cm, interpolation='nearest', cmap="jet")
    ax.figure.colorbar(im, ax=ax)  # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    # fmt = 'd'
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         if int(cm[i, j]*100 + 0.5) > 0:
    #             ax.text(j, i, format(int(round(cm[i, j]*100)) , fmt),
    #                     ha="center", va="center",
    #                     color="yellow" if cm[i, j] > thresh else "red", fontsize=5)
    fig.tight_layout()
    plt.savefig(path, dpi=400)
    # plt.show()


if __name__ == '__main__':
    # paths = ['json/CRNN_MJ.json','json/NRNC_MJ.json','json/NVNC_MJ.json','json/TRBC_MJ.json']
    #paths = ['json/CRNN.json','json/NRNC.json','json/NVNC.json','json/TRBC.json']
    paths = ['exp_json/trba_kldiv.json']
    #paths = ['json/CRNN_MJ.json','json/NRNC_MJ.json','json/NVNC_MJ.json','json/TRBC_MJ.json','json/CRNN.json','json/NRNC.json','json/NVNC.json','json/TRBC.json']
    # weight = [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]            # 加起来等于1

    data = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            data += json.load(f)

    hot = visual(data)
    
    plot_Matrix(hot, '0123456789abcdefghijklmnopqrstuvwxyz', path=f'visiual/confused_matrix/TRBA_kldiv_matrix.jpg')