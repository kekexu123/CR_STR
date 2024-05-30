import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cali_loss import *

class ParamException(Exception):
    """
    Invalid parameter exception
    """

    def __init__(self, msg, fields=None):
        self.fields = fields
        self.msg = msg

    def __str__(self):
        return self.msg

class RobustSmoothLoss(nn.Module):
    def __init__(self, opt,
                ignore_index, 
                # matric, 
                alpha=0.0, 
                exp_base=0,
                transit_time_ratio=0.2,
                normalize_length=True, 
                device=None, 
                **kwargs):
        super(RobustSmoothLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.kldiv = nn.KLDivLoss()
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha #
        # self.matric = torch.tensor(matric, dtype=torch.float32)[:-1, :-1, :-1]
        self.normalize_length = normalize_length
        self.device = device

        self.total_iterations = opt.num_iter
        self.exp_base = exp_base #
        self.counter = "iteration"
        self.epsilon = None
        self.transit_time_ratio = transit_time_ratio

        if not (self.exp_base > 0):
            error_msg = (
                "self.exp_base = "
                + str(self.exp_base)
                + ". "
                + "The exp_base has to be no less than zero"

            )
            raise (ParamException(error_msg))

        if self.counter not in ["iteration", "epoch"]:
            error_msg = (
                "self.counter = "
                + str(self.counter)
                + ". "
                + "The counter has to be iteration or epoch. "
                + "The training time is counted by eithor of them. "
            )
            raise (ParamException(error_msg))
    
    def update_epsilon_progressive_adaptive(self, pred_probs, cur_time):
        with torch.no_grad():
            # global trust/knowledge
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )

            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
            # example-level trust/knowledge
            class_num = pred_probs.shape[1]
            H_pred_probs = torch.sum(
                -(pred_probs + 1e-12) * torch.log(pred_probs + 1e-12), 1
            )
            H_uniform = -torch.log(torch.tensor(1.0 / class_num))
            example_trust = 1 - H_pred_probs / H_uniform
            # the trade-off
            self.epsilon = global_trust * example_trust
            # from shape [N] to shape [N, 1]
            self.epsilon = self.epsilon[:, None]


    def forward(self, input, target, cur_time):

        if self.counter == "epoch":
            # cur_time indicate epoch
            if not (cur_time <= self.total_epochs and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_epochs)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        else:  # self.counter == "iteration":
            # cur_time indicate iteration
            if not (cur_time <= self.total_iterations and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_iterations)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))

        # length = torch.FloatTensor(list(map(len, labels))) + 1.0
        batch_size, max_time_len, nclass = input.shape

        ce_loss = self.ce_loss(input.view(-1, input.shape[-1]), target.contiguous().view(-1))

        
        forth_target = torch.zeros_like(target)
        forth_target[:, 1:] = target[:, :-1]
        forth_target = forth_target.view(-1)
        target = target.contiguous().view(-1)
        ignore = target == self.padding_idx
        total = (ignore == True).sum().item()
        input = input.view(-1, input.shape[-1])
        log_prob = F.log_softmax(input, dim=-1)
        preds_prob = F.softmax(input, dim=1)
        
        true_dist = torch.zeros_like(input)
        true_dist.fill_(self.smoothing / (input.size(-1) - 1))
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence) #每个字符预测所属类别预测为1-alpha，其余为alpha/（所有类别数-1）

        self.update_epsilon_progressive_adaptive(preds_prob, cur_time)

        # smoothing = (1 - torch.pow((1-self.smoothing), 1.0 / length)).unsqueeze(1).repeat(1, max_time_len).view(-1)
        # weight = (smoothing.unsqueeze(1) * self.matric[forth_target.tolist(), target.tolist(), :]).to(self.device)
        
        
        # src = (1. -  weight.sum(dim=1))
        # src = src.unsqueeze(-1).repeat(1, weight.size(1)) 
        # weight.scatter_(-1, target.unsqueeze(1), src)
        # weight = weight.type_as(preds_prob)

        # denom = total if self.normalize_length else batch_size
        # loss = (-weight * log_prob).masked_fill(ignore.unsqueeze(1), 0).sum() / denom
        kl_loss = 0
        for i in range( batch_size*max_time_len):
            kl_loss_i = self.kldiv(preds_prob[i,:],true_dist[i,:])*self.epsilon[i]
            kl_loss += kl_loss_i
        
        kl_loss = kl_loss /(batch_size*max_time_len)

        
        # kl_loss =  self.kldiv(preds_prob, true_dist)


        # loss = ce_loss + self.epsilon * kl_loss
        loss = ce_loss + kl_loss

        return loss
    
class RobustSmoothLoss_v2(nn.Module):
    # 去掉example_trust部分
    def __init__(self, opt,
                ignore_index, 
                alpha=0.0, 
                exp_base=0,
                transit_time_ratio=0.2,
                device=None, 
                **kwargs):
        super(RobustSmoothLoss_v2, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.cali_loss = GraduatedLabelSmoothingAttn(ignore_index=ignore_index, alpha=alpha)
        # self.cali_loss = LabelSmoothingLoss(ignore_index=ignore_index, alpha=alpha)

        self.total_iterations = opt.num_iter
        self.exp_base = exp_base #
        self.counter = "iteration"
        self.epsilon = None
        self.transit_time_ratio = transit_time_ratio

        if not (self.exp_base > 0):
            error_msg = (
                "self.exp_base = "
                + str(self.exp_base)
                + ". "
                + "The exp_base has to be no less than zero"

            )
            raise (ParamException(error_msg))

        if self.counter not in ["iteration", "epoch"]:
            error_msg = (
                "self.counter = "
                + str(self.counter)
                + ". "
                + "The counter has to be iteration or epoch. "
                + "The training time is counted by eithor of them. "
            )
            raise (ParamException(error_msg))
    
    def update_epsilon_progressive_adaptive(self, cur_time):
        with torch.no_grad():
            # global trust/knowledge
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )

            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
            # example_trust = 1 - H_pred_probs / H_uniform
            # the trade-off
            self.epsilon = global_trust 
            # from shape [N] to shape [N, 1]
            # self.epsilon = self.epsilon[:, None]


    def forward(self, input, target, cur_time):

        if self.counter == "epoch":
            # cur_time indicate epoch
            if not (cur_time <= self.total_epochs and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_epochs)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        else:  # self.counter == "iteration":
            # cur_time indicate iteration
            if not (cur_time <= self.total_iterations and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_iterations)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))

        # length = torch.FloatTensor(list(map(len, labels))) + 1.0

        ce_loss = self.ce_loss(input.view(-1, input.shape[-1]), target.contiguous().view(-1))
        self.update_epsilon_progressive_adaptive(cur_time)
        cali_loss = self.cali_loss(input, target)
        loss = ce_loss + self.epsilon * cali_loss

        return loss
    
class PSSR_RobustSmoothLoss(nn.Module):
    def __init__(self, opt,
                 converter,
                 semantic,
                 device,
                 smooth_tail=0.,
                 ignore_index=0,
                 eos=1,
                 gamma=2,
                 alpha=0.0,
                 **kwargs
                ):
        super().__init__()
        semantic[''] = [['', '', '', '', '', ], [0.2, 0.2, 0.2, 0.2, 0.2]]
        self.converter = converter
        self.semantic = semantic
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eos = eos
        self.smooth_tail = smooth_tail
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.loss_master = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.device = device

    def update_epsilon_progressive_adaptive(self, pred_probs, cur_time):
        with torch.no_grad():
            # global trust/knowledge
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )
            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
            self.epsilon = global_trust
            # from shape [N] to shape [N, 1]
            self.epsilon = self.epsilon[:, None]

    def forward(self, preds, target, visual, labels): 
        bs =  preds.shape[0]
        loss_master = self.loss_master(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        loss_master = loss_master.view(*preds.shape[:-1]).sum(dim=-1) / (torch.tensor(list(map(len, labels))) + 1).to(self.device)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        _, preds_index = preds.max(2)

        eos_loc = torch.tensor([i.tolist().index(self.eos) if self.eos in i else (preds_index.shape[-1] + 1) for i in preds_index])
        arange = torch.arange(preds_index.shape[-1])
        mask = arange.expand(*preds_index.shape)
        mask = mask >= eos_loc.unsqueeze(-1)
        preds_max_prob[mask] = 1.0
        preds_str_prob = preds_max_prob.cumprod(dim=-1)[:, -1].detach().clone()

        smoothing_list = [visual[idx]['str'] + self.semantic[label][0] for idx, label in enumerate(labels)]
        texts, length = zip(*[self.converter.encode(texts) for texts in smoothing_list])

        loss_smooth = []
        for text, pred in zip(texts, preds):
            text = text[:, 1:]
            pred = pred.unsqueeze(0).repeat(text.shape[0], 1, 1)
            loss_smooth.append(self.loss_func(pred.view(-1, pred.shape[-1]), text.contiguous().view(-1)).unsqueeze(0))
        loss_smooth = torch.cat(loss_smooth)

        ranking = self.smooth_tail + (1.0 - self.smooth_tail) * torch.pow((1 - preds_str_prob), 2)

        loss = loss_master + self.alpha * ranking * loss_smooth

        return loss.mean()
    
class RobustSmoothLoss_ctc(nn.Module):
    def __init__(self, opt,
                alpha=0.0, 
                exp_base=0,
                transit_time_ratio=0.2,
                normalize_length=True, 
                device=None, 
                **kwargs):
        super(RobustSmoothLoss_ctc, self).__init__()
        self.ctc = nn.CTCLoss(zero_infinity=True)
        self.kldiv = nn.KLDivLoss(reduction="none")

        self.confidence = 1.0 - alpha
        self.smoothing = alpha #
        # self.matric = torch.tensor(matric, dtype=torch.float32)[:-1, :-1, :-1]
        self.normalize_length = normalize_length
        self.device = device

        self.total_iterations = opt.num_iter
        self.exp_base = exp_base #
        self.counter = "iteration"
        self.epsilon = None
        self.transit_time_ratio = transit_time_ratio

        if not (self.exp_base > 0):
            error_msg = (
                "self.exp_base = "
                + str(self.exp_base)
                + ". "
                + "The exp_base has to be no less than zero"

            )
            raise (ParamException(error_msg))

        if self.counter not in ["iteration", "epoch"]:
            error_msg = (
                "self.counter = "
                + str(self.counter)
                + ". "
                + "The counter has to be iteration or epoch. "
                + "The training time is counted by eithor of them. "
            )
            raise (ParamException(error_msg))
    
    def update_epsilon_progressive_adaptive(self, pred_probs, cur_time):
        with torch.no_grad():
            # global trust/knowledge
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )

            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
            # example-level trust/knowledge
            class_num = pred_probs.shape[2]
            H_pred_probs = torch.sum(
                -(pred_probs + 1e-12) * torch.log(pred_probs + 1e-12), 2
            )
            H_uniform = -torch.log(torch.tensor(1.0 / class_num))
            example_trust = 1 - H_pred_probs / H_uniform  # (T,B)
            #avg T
            example_trust = example_trust.mean(0)          
            # the trade-off
            self.epsilon = global_trust * example_trust   # (T,B)
            # from shape [N] to shape [N, 1]
            # self.epsilon = self.epsilon[:, None]


    def forward(self, input, labels_index, preds_size, length, cur_time):

        if self.counter == "epoch":
            # cur_time indicate epoch
            if not (cur_time <= self.total_epochs and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_epochs)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        else:  # self.counter == "iteration":
            # cur_time indicate iteration
            if not (cur_time <= self.total_iterations and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_iterations)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))

        # length = torch.FloatTensor(list(map(len, labels))) + 1.0
        batch_size, max_time_len, nclass = input.shape
        input = input.permute(1,0,2)
        preds_prob = F.softmax(input, dim=1)
        self.update_epsilon_progressive_adaptive(preds_prob, cur_time)
        
        ctc_loss = self.ctc(input, labels_index, preds_size, length)
        kl_inp = input.transpose(0, 1) #(B,T,C)
        true_dist = torch.full_like(kl_inp, 1 / input.shape[-1])
        
        kl_loss = 0
        for i in range( batch_size):
            kl_loss_i = self.kldiv(kl_inp[i,:,:],true_dist[i,:,:]).mean()*self.epsilon[i]
            kl_loss += kl_loss_i
        
        kl_loss = kl_loss /(batch_size)

        
        # kl_loss =  self.kldiv(preds_prob, true_dist)


        # loss = ce_loss + self.epsilon * kl_loss
        loss = ctc_loss+ kl_loss

        return loss