import os
import json
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import argparse
# from tools.correctness_factory import correct_factory
# from correctness_factory import correct_factory

def formatnum(x, pos):
    return '%f' % (1-np.power(10, -x))

formatter = FuncFormatter(formatnum)

def get_acc_conf(data, bin_num=100, correct_fn=None):
    N = len(data)
    n_per_bin = N//bin_num

    correct_bin = [0]*bin_num
    width_bin = [0]*bin_num
    min_p_bin = [0]*bin_num
    prob_bin = [0]*bin_num
    total_bin = [0]*bin_num

    data = sorted(data, key=lambda x: x[0])
    Brier = 0
    NLL = 0
    for n in range(N):
        if data[n][2] == data[n][3]:
            P_log = -np.log(data[n][1][:-1])
            NLL += np.sum(P_log)

    for i in range(bin_num):
        if i != bin_num-1:
            ds = data[i*n_per_bin:(i+1)*n_per_bin]
        else:
            ds = data[i*n_per_bin:]

        if correct_fn is None:
            ds_correct = np.array(list(map(lambda x: int(x[2]==x[3].lower()), ds)))
            correct_bin[i] = np.sum(ds_correct)
        else:
            ds_correct = np.array(list(map(lambda x: int(correct_fn(x[2][:-1], x[3][:-1].lower())), ds)))
            correct_bin[i] = np.sum(ds_correct)
        
        ds_conf = np.array(list(map(lambda x: x[0], ds)))
        Brier += np.sum(np.power(ds_correct-ds_conf, 2))
        
        min_p_bin[i] = ds[0][0]
        width_bin[i] = ds[-1][0] - ds[0][0]
        prob_bin[i] = sum(list(map(lambda x: x[0], ds)))
        total_bin[i] = len(ds)

    correct_bin = np.array(correct_bin)
    prob_bin = np.array(prob_bin)
    total_bin = np.array(total_bin)
    width_bin = np.array(width_bin)

    acc_bin = correct_bin/total_bin
    conf_bin = prob_bin/total_bin

    CE = np.abs(conf_bin-acc_bin)
    ECE = np.sum(CE*total_bin/np.sum(total_bin))
    ACC = np.sum(correct_bin)/np.sum(total_bin)
    Brier = Brier/np.sum(total_bin)

    return conf_bin, acc_bin, ECE, Brier

def ECE(list_data, keys=['Output'], bin_num=100, correct_fn=None, vis=False):
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
    assert len(list_data)==len(keys)
    
    #plt.figure(0, clear=True)
    #plt.figure(figsize=(8, 8))
    fig, ax = plt.subplots(dpi=350,figsize=(8, 8))
    x = np.linspace(0,1,100)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.plot(x, x,'-.', color='red',label="Expected",linewidth=2.5)
    plt.yticks( size=20)#设置大小及加粗
    plt.xticks( size=20)
    plt.xlabel('Confidence',fontsize=30)
    plt.ylabel('Accuracy',fontsize=30)
    makers = [".", "o", "v", "s", "*", "+", "p", "h", "d"]
    for i, jsonp in enumerate(list_data):
        with open(jsonp,'r') as f:
            data = json.load(f)
        conf_bin, acc_bin, ECE, Brier = get_acc_conf(data,bin_num)
        plt.plot(conf_bin, acc_bin, "b-", label=f"{keys[i]}",linewidth=2.5)
    plt.legend(  loc="upper left",#图例的位置
                 fontsize=30,
                 ncol=1,#列数
                 mode="None",#当值设置为“expend”时，图例会水平扩展至整个坐标轴区域
                 borderaxespad=0.5,#坐标轴和图例边界之间的间距
                 shadow=False,#是否为线框添加阴影
                 fancybox=True)#线框圆角处理参数
    bbox = dict(boxstyle="square", fc="#75bbfd",alpha=0.4)
    ax.annotate(text=f"ACE={ECE*100:0.2f}%",xy=(0.6,0.6) ,xytext=(250, 30),
            xycoords='axes points',fontsize=34,bbox=bbox
            )
    plt.tight_layout()
    plt.savefig('dcss_sta.jpg')

    # return ECE

if __name__ == '__main__':

    file_name = [
        '/home/mdisk2/xukeke/CR_STR/exp_json/dcss_sta.json',
    ]

    # file_name = [
    #     '../deep-text-recognition-benchmark/TPS-ResNet-None-Attn_data.json',
    #     '../deep-text-recognition-benchmark/TPS-ResNet-None-Attn_LS.json',
    #     '../deep-text-recognition-benchmark/TPS-ResNet-None-Attn_SLS.json',
    #     '../deep-text-recognition-benchmark/TPS-ResNet-None-Attn_CALS.json',
    #     'TRNA_TS_1.3.json'
    # ]

    # file_name = [
    #     'ASTER_blank_data.json',
    #     'CMN_train_ASTER_LS.json',
    #     'CMN_train_ASTER_SLS.json',
    #     'CMN_train_ASTER_CALS.json',
    #     # 'CMN_train_ASTER_TS_1.1.json',
    #     # 'CMN_train_ASTER_TS_1.2.json',
    #     'CMN_train_ASTER_TS_1.3.json'
    #     # 'CMN_train_ASTER_TS_1.35.json',
    #     # 'CMN_train_ASTER_TS_1.4.json'
    # ]

    # with open(file_name,'r') as f:
        # data = json.load(f)

    ECE(file_name, bin_num=15, correct_fn=None, vis=True)
    # ECE(data, 15, None, vis=True)
    # ECE(data, 100, correct_fn=correct_factory('ECE1'), vis=True)
    # ECE(data, 100, correct_fn=correct_factory('ECE2'),vis=True)


