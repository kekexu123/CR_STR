import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import pandas as pd

#sns.set_style("darkgrid")
from matplotlib import rcParams

# rcParams['font.family'] = 'Arial'
# from calibration import DeSequence, ACE
from get_op_list import get_op_seq

rcParams['font.size'] = 60

def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
    """Collects predictions into bins used to draw a reliability diagram.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins

    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.

    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.

    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert (len(confidences) == len(pred_labels))
    assert (len(confidences) == len(true_labels))
    assert (num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return {"accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bins": bins,
            "avg_accuracy": avg_acc,
            "avg_confidence": avg_conf,
            "expected_calibration_error": ece,
            "max_calibration_error": mce}

def _confidence_histogram_subplot(ax, bin_data,
                                  draw_averages=True,
                                  title="Examples per bin",
                                  xlabel="Confidence",
                                  ylabel="Count"):
    """Draws a confidence histogram into a subplot."""
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size / 2.0

    ax.bar(positions, counts, width=bin_size * 0.9)

    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.set_ylabel(ylabel)

    if draw_averages:
        acc_plt = ax.axvline(x=bin_data["avg_accuracy"], ls="solid", lw=5,
                             c="red", label="Accuracy")
        conf_plt = ax.axvline(x=bin_data["avg_confidence"], ls="dotted", lw=5,
                              c="black", label="Avg. confidence")
        ax.legend(handles=[acc_plt, conf_plt])

    ax.grid(True, which='major', linestyle='--', linewidth=0.5)

def DeSequence(data):
    '''
    data: [
        [
            word_confidence, 
            [char_conf1, char_conf2,...,char_conf3],
            pred_str,
            gt_str,
        ], 
        ...
    ]
    bin_num: Num of Bin to calculate ECE
    '''
    nclass=37
    Deseq_data_head = []
    Deseq_data_tail = []
    head_list = ['e', 'a', 'o', 's', 't', 'r', 'i', 'n', 'l', 'c']
    for i,d in enumerate(data):
        char_conf, pred, gt = d[1:]
        op_str = get_op_seq(pred, gt)
        gt, pred, op_str = list(map(list, [gt, pred, op_str]))
        if len(char_conf) == len(pred):
            # pass
        # assert(len(char_conf) == len(pred))

            for op in op_str:
                if op == 's':
                    gg = gt.pop(0)
                    pp = pred.pop(0)
                    dd = [char_conf.pop(0), 0, pp, gg]
                elif op == '#':
                    gg = gt.pop(0)
                    pp = pred.pop(0)
                    dd = [char_conf.pop(0), 1, pp, gg]
                elif op == 'd':
                    gg = '#'
                    pp = pred.pop(0)
                    dd = [char_conf.pop(0), 0, pp, gg]
                else:
                    gg = gt.pop(0)
                    pp = '#'
                    dd = [1/nclass, 0, pp, gg]
                
                if gg in head_list:
                    Deseq_data_head.append(dd) 
                elif gg not in head_list and gg != '#':
                    Deseq_data_tail.append(dd) 

    
    return Deseq_data_head, Deseq_data_tail

def plot_hist(bin_data, file_name):
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(20,20), dpi=600)
    for i in ['top', 'bottom', 'right', 'left']:
        ax.spines[i].set_color('black')
        ax.spines[i].set_linewidth(5.0)
    plt.tight_layout()
    _confidence_histogram_subplot(ax, bin_data, title="")
    plt.savefig(file_name)
    plt.close


if __name__ == '__main__':
    #------------------hist diagram ----------------
    with open('/home/mdisk2/xukeke/CR_STR/exp_json/baseline_all.json') as f:
        pred = json.load(f)

    # pred = np.array(pred)
    # confidence = [item[0] for item in pred]  # 置信度
    # preds = [item[-2] for item in pred]  # 预测标签
    # gt = [item[-1] for item in pred]  # 真实标签
    # confidence = np.array(confidence)
    # preds = np.array(preds)
    # gt = np.array(gt)
    # confidence, preds, gt = pred[:, 0], pred[:, -2], pred[:, -1]

    data_head, data_tail = DeSequence(pred)

    confidence_head = np.array([item[0] for item in data_head])
    preds_head = np.array([item[-2] for item in data_head])
    gt_head = np.array([item[-1] for item in data_head])
    bin_data_head = compute_calibration(true_labels=gt_head, pred_labels=preds_head, confidences=confidence_head, num_bins=15)

    confidence_tail = np.array([item[0] for item in data_tail])
    preds_tail = np.array([item[-2] for item in data_tail])
    gt_tail = np.array([item[-1] for item in data_tail])
    bin_data_tail = compute_calibration(true_labels=gt_tail, pred_labels=preds_tail, confidences=confidence_tail, num_bins=15)

    plot_hist(bin_data_head, 'dcss_head_hist.jpg')
    plot_hist(bin_data_tail, 'dcss_tail_hist.jpg')
    