import os
import sys
import time
import argparse
import re
from datetime import date

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import editdistance as ed

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from tools.dataset_test import hierarchical_dataset, AlignCollate, select_dataset
from tools.ctc_utils import ctc_prefix_beam_search
from modules.calibrators import  ModelWithTemperature
from tools.calibration import ACE,ECE
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def geometric_mean(data):  # 计算几何平均
    total = 1
    for i in data:
        total *= i
    return torch.pow(total, 1/len(data))

def arithmetic_mean(data):  # 计算算术平均
    log_data = torch.log(data)
    return torch.exp(log_data.sum()/len(data))

def benchmark_all_eval(model,
                       criterion,
                       converter,
                       opt,
                       save_path,
                       calculate_infer_time=False):

    if opt.eval_type == 'benchmark':
        """ evaluation with 6 benchmark evaluation datasets """
        eval_data_list = [
            'IC15_2077', 'IIIT5k_3000', 'IC13_857', 'IC13_1015', 'SVTP', 'CUTE80', 'IC15_1811', 'SVT'
            # 'CUTE80', 'IC13_1015', 'IC13_857', 'IC15_1811', 'IC15_2077', 'IIIT5k_3000', 'SVT', 'SVTP'
            # 'IIIT5k_3000', 'SVT', 'IC13_857', 'IC13_1015', 'IC15_1811','IC15_2077', 'SVTP', 'CUTE80'
            # 'IIIT5k_3000', 'SVT', 'IC13_1015','IC15_2077', 'SVTP', 'CUTE80'
            # 'IC15_1811'
        ]

    elif opt.eval_type == 'addition':
        """ evaluation with 7 additionally collected evaluation datasets """
        eval_data_list = [
            '5.COCO', '6.RCTW17', '7.Uber', '8.ArT', '9.LSVT', '10.MLT19',
            '11.ReCTS'
        ]
    elif opt.eval_type == "simple":
        eval_data_list = [
            'IIIT5k_3000', 'IC15_1811', 'IC13_857', 'CUTE80', 'SVTP', 'SVT'
        ]
    # elif opt.eval_type == 'ic13_ic15':
    elif opt.eval_type == 'validation':
        eval_data_list = ['IC13_trainLmdb', 'IC15_trainLmdb']
    
    elif opt.eval_type == 'valid':
        eval_data_list = ['1.SVT', '2.IIIT','3.IC13','4.IC15','5.COCO','6.RCTW17','7.Uber','8.ArT','9.LSVT','10.MLT19','11.ReCTS']
    
    elif opt.eval_type == 'ic15_1811':
        eval_data_list = ['IC15_1811']

    elif opt.eval_type == 'label':
        eval_data_list = ['CVPR2016', 'NIPS2014']

    if calculate_infer_time:
        eval_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        eval_batch_size = opt.batchSize

    accuracy_list = []
    dataset_pred_list = []
    total_forward_time = 0
    total_eval_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{save_path}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(f'{opt.eval_data}/{opt.eval_type}', eval_data)
        # eval_data_path = os.path.join(f'{opt.eval_data}', eval_data)
        AlignCollate_eval = AlignCollate(opt)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path,
                                                        opt=opt)
        # eval_data, eval_data_log = select_dataset(root=eval_data_path,
        #                                                  opt=opt)
        eval_loader = torch.utils.data.DataLoader(eval_data,
                                                  batch_size=eval_batch_size,
                                                  shuffle=False,
                                                  num_workers=int(opt.workers),
                                                  collate_fn=AlignCollate_eval,
                                                  pin_memory=True)

        _, accuracy_by_best_model, _, _, _, infer_time, length_of_data, preds_data,_ = validation(
            model, criterion, eval_loader, converter, opt, tqdm_position=0)
        accuracy_list.append(f'{accuracy_by_best_model:0.2f}')
        dataset_pred_list = dataset_pred_list + preds_data
        total_forward_time += infer_time
        total_eval_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        log.write(eval_data_log)
        print(f'Acc {accuracy_by_best_model:0.2f}')
        log.write(f'Acc {accuracy_by_best_model:0.2f}\n')
        print(dashed_line)
        log.write(dashed_line + '\n')

    import json
    if opt.json_path:
        with open(opt.json_path,'w') as f: 
            f.write(json.dumps(dataset_pred_list, ensure_ascii=False))  
    else:
        with open(f'{opt.checkpoint_root}/{save_path}/all_dataset_preds.json','w') as f: 
            f.write(json.dumps(dataset_pred_list, ensure_ascii=False))  
    current_ace, acc, _, pzhECE, flag_str = ACE(dataset_pred_list, bin_num=15, vis=True, correct_fn=None, testing=True)
    current_ece, _, _ = ECE(dataset_pred_list, bin_num=15, correct_fn=None)
    print(current_ace, current_ece)
    averaged_forward_time = total_forward_time / total_eval_data_number * 1000
    total_accuracy = total_correct_number / total_eval_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    eval_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, accuracy_list):
        eval_log += f'{name}: {accuracy}\t'
    eval_log += f'total_accuracy: {total_accuracy:0.2f}\t'
    eval_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.2f}'
    print(eval_log)
    log.write(eval_log + '\n')

    # for convenience
    print('\t'.join(accuracy_list))
    print(f'Total_accuracy:{total_accuracy:0.2f}')
    log.write('\t'.join(accuracy_list) + '\n')
    log.write(f'Total_accuracy:{total_accuracy:0.2f}' + '\n')
    log.close()

    # for convenience
    today = date.today()
    log_all_model = open(f'./evaluation_log/log_multiple_test_{today}.txt',
                         'a')
    log_all_model.write('\t'.join(accuracy_list) + '\n')
    log_all_model.close()

    return total_accuracy, eval_data_list, accuracy_list


def validation(model, criterion, eval_loader, converter, opt, tqdm_position=1):
    """ validation or evaluation """
    n_correct = 0
    after_select_num = 0
    label_error_num = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    preds_data = []
    # T = 0
    for i, (image_tensors, labels) in enumerate(eval_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        labels_index, labels_length = converter.encode(
            labels, batch_max_length=opt.batch_max_length)

        if 'CTC' in opt.Prediction:
            start_time = time.time()
            preds = model(image)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            cost = criterion(
                preds.log_softmax(2).permute(1, 0, 2), labels_index,
                preds_size, labels_length)

        else:
            text_for_pred = torch.LongTensor(batch_size).fill_(
                converter.dict['[SOS]']).to(device)

            start_time = time.time()
            # T = 0.8
            with torch.no_grad():
                preds = model(image, text_for_pred, is_train=False)
                # preds_tensor = preds.view(batch_size, preds.size(1) * preds.size(2))
                # preds_tensor = preds_tensor.unsqueeze(2).unsqueeze(3).expand(-1, -1, image.size(2), image.size(3))
                # T = ltsmodel(preds_tensor)
                # preds = preds/T
            forward_time = time.time() - start_time

            target = labels_index[:, 1:]  # without [SOS] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]),
                             target.contiguous().view(-1))

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_size = torch.IntTensor([preds.size(1)] *
                                     preds_index.size(0)).to(device)
        preds_str = converter.decode(preds_index, preds_size)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []

        for gt, prd, prd_max_prob, logit in zip(labels, preds_str, preds_max_prob, preds):
            if 'Attn' in opt.Prediction:
                prd_EOS = prd.find('[EOS]')
                prd = prd[:
                          prd_EOS]  # prune after "end of sentence" token ([EOS])
                prd_max_prob = prd_max_prob[:prd_EOS]
                # calculate confidence score (= multiply of prd_max_prob)
                try:               
                    confidence_score = prd_max_prob.cumprod(dim=0)[-1] # 累乘
                except:
                    confidence_score = torch.tensor(0)
                token_confs = prd_max_prob.cpu().numpy().tolist()

            if 'CTC' in opt.Prediction:
                hyps, confidence_score = ctc_prefix_beam_search(logit.unsqueeze(0), beam_size=1)
                confidence_score = confidence_score[0]
                prd = "".join([converter.idict[i] for i in hyps[0][0]])
                token_confs = []
            """
            In our experiment, if the model predicts at least one [UNK] token, we count the word prediction as incorrect.
            To not take account of [UNK] token, use the below line.
            prd = prd.replace('[UNK]', '') 
            """

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting. = same with ASTER
            if not opt.sensitive:
                gt = gt.lower()
                prd = prd.lower()
            out_of_alphanumeric_case_insensitve = f'[^{opt.character}]'
            gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)
            prd = re.sub(out_of_alphanumeric_case_insensitve, '', prd)

            if opt.NED:
                # ICDAR2019 Normalized Edit Distance
                if len(gt) == 0 or len(prd) == 0:
                    norm_ED += 0
                elif len(gt) > len(prd):
                    norm_ED += 1 - ed(prd, gt) / len(gt)
                else:
                    norm_ED += 1 - ed(prd, gt) / len(prd)

            else:
                if prd == gt:
                    n_correct += 1

            confidence_score_list.append(confidence_score)
            # preds_data.append([confidence_score.cpu().numpy().tolist(), token_confs, prd, gt])
            preds_data.append([confidence_score.cpu().numpy().tolist(), token_confs, prd, gt])
            # 增加伪标签筛选并计算label_error_rate部分
            if confidence_score>=opt.confident_threshold:
                after_select_num +=1
                if gt != prd:
                    label_error_num +=1

    if opt.NED:
        # ICDAR2019 Normalized Edit Distance. In web page, they report % of norm_ED (= norm_ED * 100).
        score = norm_ED / float(length_of_data) * 100
    else:
        score = n_correct / float(length_of_data) * 100  # accuracy

    if after_select_num != 0:
        label_error_rate = label_error_num / float(after_select_num) * 100 # 标签噪声率
    else:
        label_error_rate = 0.0

    return valid_loss_avg.val(
    ), score, preds_str, confidence_score_list, labels, infer_time, length_of_data, preds_data, label_error_rate

def validation_lts(model,ltsmodel, criterion, eval_loader, converter, opt):
    preds_all = []
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    preds_data = []
    T_list = []
    # 先求T
    for i, (image_tensors, labels) in enumerate(eval_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        labels_index, labels_length = converter.encode(
            labels, batch_max_length=opt.batch_max_length)

        if 'CTC' in opt.Prediction:
            start_time = time.time()
            preds = model(image)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)

        else:
            text_for_pred = torch.LongTensor(batch_size).fill_(
                converter.dict['[SOS]']).to(device)

            start_time = time.time()
            with torch.no_grad():
                preds = model(image, text_for_pred, is_train=False)
                # 先拉成(B,T * C)
                preds_tensor = preds.view(preds.size(0), preds.size(1) * preds.size(2))
                # 再扩展为（B,T * C，H, W）
                preds_tensor = preds_tensor.unsqueeze(2).unsqueeze(3).expand(-1, -1, opt.imgH, opt.imgW)
                T = ltsmodel(preds_tensor)
                T_list.append(T)
            # preds_list.append(preds)
            forward_time = time.time() - start_time
    # add by xkk
    T = sum(T_list) / len(T_list)
    # 再得出校准后的预测
    for i, (image_tensors, labels) in enumerate(eval_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        labels_index, labels_length = converter.encode(
            labels, batch_max_length=opt.batch_max_length)

        if 'CTC' in opt.Prediction:
            start_time = time.time()
            preds = model(image)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)

        else:
            text_for_pred = torch.LongTensor(batch_size).fill_(
                converter.dict['[SOS]']).to(device)

            start_time = time.time()
            with torch.no_grad():
                preds = model(image, text_for_pred, is_train=False)
                preds = preds/T
            forward_time = time.time() - start_time

            target = labels_index[:, 1:]  # without [SOS] Symbol

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_size = torch.IntTensor([preds.size(1)] *
                                     preds_index.size(0)).to(device)
        preds_str = converter.decode(preds_index, preds_size)

        infer_time += forward_time

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, prd, prd_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                prd_EOS = prd.find('[EOS]')
                prd = prd[:
                          prd_EOS]  # prune after "end of sentence" token ([EOS])
                prd_max_prob = prd_max_prob[:prd_EOS]
            """
            In our experiment, if the model predicts at least one [UNK] token, we count the word prediction as incorrect.
            To not take account of [UNK] token, use the below line.
            prd = prd.replace('[UNK]', '') 
            """

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting. = same with ASTER
            gt = gt.lower()
            prd = prd.lower()
            alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
            out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
            gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)
            prd = re.sub(out_of_alphanumeric_case_insensitve, '', prd)

            if opt.NED:
                # ICDAR2019 Normalized Edit Distance
                if len(gt) == 0 or len(prd) == 0:
                    norm_ED += 0
                elif len(gt) > len(prd):
                    norm_ED += 1 - ed(prd, gt) / len(gt)
                else:
                    norm_ED += 1 - ed(prd, gt) / len(prd)

            else:
                if prd == gt:
                    n_correct += 1

            # calculate confidence score (= multiply of prd_max_prob)
            try:               
                confidence_score = prd_max_prob.cumprod(dim=0)[-1] # 累乘

            except:
                # confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([EOS])
                confidence_score = torch.tensor(0)
            token_confs = prd_max_prob.tolist()

            confidence_score_list.append(confidence_score)
            preds_data.append([confidence_score.cpu().numpy().tolist(), token_confs, prd, gt])

    return preds_data, T

def validation_ts(model, eval_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    preds_data = []
    T_list = []
    ts_model = ModelWithTemperature(model, converter)
    # Tune the model temperature, and save the results
    T = ts_model.set_temperature(eval_loader)
# 再得出校准后的预测
    for i, (image_tensors, labels) in enumerate(eval_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)

        if 'CTC' in opt.Prediction:
            start_time = time.time()
            preds = model(image)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)

        else:
            text_for_pred = torch.LongTensor(batch_size).fill_(
                converter.dict['[SOS]']).to(device)

            start_time = time.time()
            with torch.no_grad():
                preds = model(image, text_for_pred, is_train=False)
                preds = preds/T
            forward_time = time.time() - start_time

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_size = torch.IntTensor([preds.size(1)] *
                                     preds_index.size(0)).to(device)
        preds_str = converter.decode(preds_index, preds_size)

        infer_time += forward_time

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, prd, prd_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                prd_EOS = prd.find('[EOS]')
                prd = prd[:
                          prd_EOS]  # prune after "end of sentence" token ([EOS])
                prd_max_prob = prd_max_prob[:prd_EOS]
            """
            In our experiment, if the model predicts at least one [UNK] token, we count the word prediction as incorrect.
            To not take account of [UNK] token, use the below line.
            prd = prd.replace('[UNK]', '') 
            """

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting. = same with ASTER
            gt = gt.lower()
            prd = prd.lower()
            alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
            out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
            gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)
            prd = re.sub(out_of_alphanumeric_case_insensitve, '', prd)

            if opt.NED:
                # ICDAR2019 Normalized Edit Distance
                if len(gt) == 0 or len(prd) == 0:
                    norm_ED += 0
                elif len(gt) > len(prd):
                    norm_ED += 1 - ed(prd, gt) / len(gt)
                else:
                    norm_ED += 1 - ed(prd, gt) / len(prd)

            else:
                if prd == gt:
                    n_correct += 1

            # calculate confidence score (= multiply of prd_max_prob)
            try:               
                confidence_score = prd_max_prob.cumprod(dim=0)[-1] # 累乘

            except:
                # confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([EOS])
                confidence_score = torch.tensor(0)
            token_confs = prd_max_prob.tolist()

            confidence_score_list.append(confidence_score)
            preds_data.append([confidence_score.cpu().numpy().tolist(), token_confs, prd, gt])

    return preds_data, T
