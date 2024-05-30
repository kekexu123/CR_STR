import json
import copy
import numpy as np
from matplotlib import pyplot as plt
from get_op_list import get_op_seq


def visual(data, classes='0123456789abcdefghijklmnopqrstuvwxyz'):
    hot = np.zeros((len(classes), len(classes)))
    classes = list(classes) + ['-']
    matric = np.zeros((len(classes), len(classes)))

    template = {i: [] for i in classes[:-1]}
    conf_dict = {i: copy.deepcopy(template) for i in classes[:-1]}
    alphabets='abcdefghijklmnopqrstuvwxyz'
    numbers='0123456789'
    alphabet =0
    number = 0
    total_num =0

    for d in data:
        tokens_conf, pred, gt = d[1:]
        for ii in gt:
            total_num +=1
            if ii in alphabets:
                alphabet +=1
            elif ii in numbers:
                number += 1
        if gt == pred or len(gt) != len(pred): continue
        op_str = get_op_seq(pred, gt)
        gt, pred, op_str = list(map(list, [gt, pred, op_str]))
        pointer = 0
        for op in op_str:
            if op == 's':
                gg = gt.pop(0)
                pp = pred.pop(0)
                hot[classes.index(gg)][classes.index(pp)] += 1
                conf_dict[gg][pp].append(tokens_conf[pointer])
                pointer += 1
            elif op == '#':
                gg = gt.pop(0)
                pp = pred.pop(0)
                # conf_dict[gg][pp].append(tokens_conf[pointer])
                pointer += 1
            elif op == 'd':
                gg = '-'
                pp = pred.pop(0)
                pointer += 1
            else:
                gg = gt.pop(0)
                pp = '-'
            matric[classes.index(gg)][classes.index(pp)] += 1

    print('the percentage of numbers in gt is '+ str((number*100)/total_num))
    print('the percentage of characters in gt is '+ str((alphabet*100)/total_num))

    for i in range(len(classes) - 1):
        hot[i, :] = hot[i, :] / (hot[i, :].sum() + 1e-16)
    for i in range(len(classes)):
        matric[i, :] = matric[i, :] / (matric[i, :].sum() + 1e-16)

    return hot, matric.diagonal().tolist(), conf_dict


def plot_Matrix(cm, results,classes, title=None, cmap=plt.cm.Blues, path='confuse_matric.jpg'):
    plt.rc('font', family='Times New Roman', size='8')  # 设置字体样式、大小

    # 按行进行归一化
    cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 10e-6)
    # print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    cm_copy = cm.copy()
    top10 = []
    classes_list = list(classes)
    classes_delete = list(classes)
    # for row in str_cm:
    #     print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         if int(cm[i, j]*100 + 0.5) == 0:
    #             cm[i, j]=0
    print('o-0 frequency is'+ str(cm_copy[24][0]*100))
    for i in range(10):
        gt_idx, pred_idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
        top10.append(classes[gt_idx])
        # print(classes[gt_idx], classes[pred_idx], np.max(cm_copy))
        cm_copy[gt_idx, pred_idx] = 0
    # print(top10)

    for k,v in results.items():
        if k in top10:
            print(k, v)

    for item in classes_list[::-1]:
        if item not in top10:
            if item != 'o':
                del str_cm[classes_list.index(item)]
                classes_delete.remove(item)
    print(classes_delete)

    '''for k,v in results.items():
        if k == '3':
            print(k, v)
        elif k == '9':
            print(k,v)'''

    fig, ax = plt.subplots()
    str_cm = np.array(str_cm,dtype=np.float32)
    # im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    im = ax.imshow(str_cm, interpolation='nearest', cmap="jet")
    ax.figure.colorbar(im, ax=ax)  # 侧边的颜色条带

    ax.set(xticks=np.arange(str_cm.shape[1]),
           yticks=np.arange(str_cm.shape[0]),
           xticklabels=classes, yticklabels=classes_delete,
           title='Attn',
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(str_cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(str_cm.shape[0] + 1) - .5, minor=True)
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
    lang_dict = '0123456789abcdefghijklmnopqrstuvwxyz'
    paths = ['json/MASTER-mj.json','json/MASTER.json']
    #paths = ['json/NVNC_MJ.json','json/NVNC.json']
    save_path = f'Confused_matrix/attn-matrix_o.jpg'

    data = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            data += json.load(f)

    hot, _, confs_dict = visual(data)
    results = {}
    for idx, gt in enumerate(lang_dict):
        max_len = 0
        flag = ''
        for k,v in confs_dict[gt].items():
            if len(v) > max_len:
                max_len = len(v)
                flag = k
                freq = hot[idx][lang_dict.index(k)]
                avg_conf = sum(v) / len(v)
        results[gt] = [flag, freq*100, avg_conf*100]
    plot_Matrix(hot,results, lang_dict, path=save_path)

    
    xx=0
