import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import json

from model import Model
from .cali_loss import ClassAwareLablesmoothing_v1, LabelAwareSmoothing, ClassAwareLablesmoothing_v2, ClassAwareLablesmoothing_v3, ClassAwareLablesmoothing_v4
from .loss_coral import coral_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_pred_to_pseudo_parallel(targets_u, converter):
    blank = converter.dict[' ']
    unl_N, unl_len = targets_u.shape #(num,len)   B T 
    targets_u = targets_u.view(unl_N, -1)    # [B T] -> [B T]
    indexs = torch.arange(0, unl_len).repeat(unl_N, 1).to(device) #(num,len) [0,...,len]*num
    indexs[targets_u > blank] += 10000
    eos_index = indexs.argmin(
        dim=-1)  # find smallest index of extend word(eos bos pad blank) 找到每个样本的最小索引值
    eos_index[(eos_index == 0) & (targets_u[:, 0] > blank)] = unl_len - 1
    unl_length = eos_index + 1
    new_eos_index = eos_index.expand(unl_len, unl_N).permute(1, 0) # B T
    indexs = torch.arange(0, unl_len).expand(unl_N, unl_len).to(device)   # B T
    pad_mask = (indexs - new_eos_index) > 0
    non_pad_mask = (pad_mask == False)

    cur_indexs = torch.arange(0, unl_N).to(device)
    targets_u[cur_indexs, eos_index] = converter.dict['[EOS]']
    targets_u[pad_mask] = converter.dict['[PAD]']

    # convert prediction to pseudo label
    pseudo = torch.cuda.LongTensor(unl_N, unl_len + 1) # B T+1 (+sos)
    pseudo[:, 0] = converter.dict['[SOS]']
    pseudo[:, 1:] = targets_u
    return pseudo, non_pad_mask, unl_length

def convert_pred_to_pseudo_parallel_ctc(targets_u, converter):
    blank = converter.dict[' ']
    pad = converter.dict['[PAD]']
    unl_N, unl_len = targets_u.shape #(num,len)
    targets_u = targets_u.view(unl_N, -1)
    char_index = torch.LongTensor(unl_N, unl_len).fill_(pad)   
    char_len_list = [] 
    unl_length = []
    indexs = torch.arange(0, unl_len).repeat(unl_N, 1).to(device) #(num,len) [0,...,len]*num
    for n in range(unl_N):
        word_index = []
        for i in range(unl_len):
            if targets_u[n, i] > blank and not (i > 0 and targets_u[n, i] == targets_u[n, i - 1]):
                word_index.append(targets_u[n,i])  # 合并删除后的index
        word_index = torch.LongTensor(word_index).to(device)
        char_len = len(word_index)  #  where begin to pad
        char_len_list.append(char_len-1) # pad begin index
        unl_length.append(char_len) # pseudo label length only char
        char_index[n,:char_len] = word_index
    char_len_list = torch.LongTensor(char_len_list).to(device)
    unl_length = torch.Tensor(unl_length).to(dtype=torch.int32).to(device)
    new_index = char_len_list.expand(unl_len, unl_N).permute(1, 0)
    pad_mask = (indexs - new_index) > 0
    non_pad_mask = (pad_mask == False)
    # convert prediction to pseudo label
    pseudo = torch.cuda.LongTensor(unl_N, unl_len)
    pseudo = char_index.to(device)
    

    return pseudo, non_pad_mask, unl_length


class CrossEntropyLoss(nn.Module):

    def __init__(self, opt, online_for_init_target, converter):
        super(CrossEntropyLoss, self).__init__()

        self.opt = opt
        self.converter = converter
        self.target_model = Model(opt)  # create the ema model
        self.target_model = torch.nn.DataParallel(self.target_model).to(device)
        self.target_model.train()
        with open('datasets/char_67_num.json') as f:
            cls_num_list = json.load(f)
        num_k = min(cls_num_list)
        cls_num_list = [num_k]*5 + cls_num_list
        head_cls = list(np.argsort(cls_num_list)[-10:][::-1])
        if opt.calibrator == 'LAS': 
            self.ce_criterion = LabelAwareSmoothing(ignore_index=converter.dict['[PAD]'], cls_num_list=cls_num_list, alpha=opt.alpha).to(device)
        elif opt.calibrator == 'CAL':
            # self.ce_criterion = ClassAwareLablesmoothing_v1(ignore_index=converter.dict['[PAD]'], cls_num_list=cls_num_list, alpha=opt.alpha, head_cls=head_cls).to(device)
            # self.ce_criterion = ClassAwareLablesmoothing_v2(ignore_index=converter.dict['[PAD]'], cls_num_list=cls_num_list, alpha=opt.alpha, head_cls=head_cls).to(device)
            # self.ce_criterion = ClassAwareLablesmoothing_v3(ignore_index=converter.dict['[PAD]'], cls_num_list=cls_num_list, alpha=opt.alpha, head_cls=head_cls).to(device)
            self.ce_criterion = ClassAwareLablesmoothing_v4(ignore_index=converter.dict['[PAD]'], cls_num_list=cls_num_list, alpha=opt.alpha, head_cls=head_cls).to(device)
        else:
            self.ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=converter.dict['[PAD]']).to(device)
        if opt.Prediction == 'Attn':
            self.decoder = 'Attn'
            self.text_for_pred = torch.LongTensor(opt.batchSize).fill_(
                opt.sos_token_index).to(device)
        else: 
            self.decoder = 'CTC'

        # copy online and init
        for param_t, param_s in zip(self.target_model.parameters(),
                                    online_for_init_target.parameters()):
            param_t.data.copy_(param_s.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

        for buffer_t, buffer_s in zip(self.target_model.buffers(),
                                      online_for_init_target.buffers()):
            buffer_t.copy_(buffer_s)

    def _update_ema_variables(self, online, iteration, alpha=0.999):
        # Use the true average until the exponential average is more correct
        # alpha = min(1 - 1 / (iteration + 1), alpha)
        for param_t, param_s in zip(self.target_model.parameters(),
                                    online.parameters()):
            param_t.data = param_t.data * alpha + param_s.data * (1. - alpha)
        for buffer_t, buffer_s in zip(self.target_model.buffers(),
                                      online.buffers()):
            buffer_t.copy_(buffer_s)

    def forward(self,
                unl_img,
                aug_img,
                online,
                iteration,
                total_iter,
                l_local_feat=None,
                l_logit=None,
                l_text=None):
        loss_SemiSL = None
        l_da = None
        l_confident_ratio = 0
        self._update_ema_variables(online, iteration, self.opt.ema_alpha)
        self.target_model.eval()
        if self.decoder == 'Attn':
            with torch.no_grad():
                unl_logit_weak = self.target_model(unl_img,
                                                text=self.text_for_pred,
                                                is_train=False)
        else:
            with torch.no_grad():
                unl_logit_weak = self.target_model(unl_img)

        self.target_model.train()
        _, preds_index = unl_logit_weak.max(2)
        sequence_score, preds_index = unl_logit_weak.softmax(dim=-1).max(
            dim=-1)
        targets_u, non_pad_mask, unl_length = convert_pred_to_pseudo_parallel(
            preds_index, self.converter)
        sequence_score[non_pad_mask == False] = 10000
        batch_score = sequence_score.min(dim=-1)[0]
        mask = batch_score.ge(self.opt.confident_threshold)
        confident_ratio = mask.to(torch.float).mean()
        if iteration % 100 == 0:
            print('score', batch_score[:2])
            print('targets_u', targets_u[:2])
        if confident_ratio > 0:
            unl_logit_strong, unl_local_feats = online(aug_img,
                                                       targets_u[:, :-1],
                                                       use_project=True,
                                                       return_local_feat=True)
            unl_score, unl_index = unl_logit_strong.log_softmax(dim=-1).max(
                dim=-1)
            unl_pred, non_pad_mask, _ = convert_pred_to_pseudo_parallel(
                unl_index, self.converter)
            unl_pred = unl_pred[:, 1:]
            unl_score[non_pad_mask == False] = 0
            unl_prob = unl_score.sum(dim=-1).exp()
            unl_mask = unl_prob.ge(self.opt.confident_threshold)

            l_score, l_index = l_logit.log_softmax(dim=-1).max(dim=-1)
            l_pred, non_pad_mask, _ = convert_pred_to_pseudo_parallel(
                l_index, self.converter)
            l_pred = l_pred[:, 1:]
            l_score[non_pad_mask == False] = 0
            l_prob = l_score.sum(dim=-1).exp()
            l_mask = l_prob.ge(self.opt.l_confident_threshold)

            l_confident_ratio = l_mask.to(torch.float).mean()
            s_confident_ratio = unl_mask.to(torch.float).mean()
            if l_confident_ratio > 0 and s_confident_ratio > 0:
                l_da = coral_loss(l_local_feat[l_mask],
                                  l_pred[l_mask],
                                  unl_local_feats[unl_mask],
                                  unl_pred[unl_mask],
                                  BLANK=self.converter.dict[' '])
                if l_da is not None:
                    l_da = l_da * self.opt.lambda_mmd
            current_logit_strong = unl_logit_strong[mask].view(
                -1, unl_logit_strong.size(-1))
            current_target = targets_u[:, 1:][mask].view(-1)
            loss_SemiSL = confident_ratio * self.ce_criterion(
                current_logit_strong, current_target)
            loss_SemiSL = self.opt.lambda_cons * loss_SemiSL
        return loss_SemiSL, confident_ratio, l_da, l_confident_ratio, batch_score


class KLDivLoss(nn.Module):

    def __init__(self, opt, online_for_init_target, converter):
        super(KLDivLoss, self).__init__()

        self.opt = opt
        self.converter = converter
        self.target_model = Model(opt)  # create the ema model
        self.target_model = torch.nn.DataParallel(self.target_model).to(device)
        self.target_model.train()
        self.kldiv_criterion = torch.nn.KLDivLoss(
            reduction='batchmean').to(device)
        if opt.Prediction == 'Attn':
            self.text_for_pred = torch.LongTensor(opt.batchSize).fill_(
                opt.sos_token_index).to(device)
        else:
            self.text_for_pred = torch.LongTensor(opt.batchSize).fill_(0).to(device)
    

        # copy online and init
        for param_t, param_s in zip(self.target_model.parameters(),
                                    online_for_init_target.parameters()):
            param_t.data.copy_(param_s.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

        for buffer_t, buffer_s in zip(self.target_model.buffers(),
                                      online_for_init_target.buffers()):
            buffer_t.copy_(buffer_s)

    def _update_ema_variables(self, online, iteration, alpha=0.999):
        # Use the true average until the exponential average is more correct
        # alpha = min(1 - 1 / (iteration + 1), alpha)
        for param_t, param_s in zip(self.target_model.parameters(),
                                    online.parameters()):
            param_t.data = param_t.data * alpha + param_s.data * (1. - alpha)
        for buffer_t, buffer_s in zip(self.target_model.buffers(),
                                      online.buffers()):
            buffer_t.copy_(buffer_s)

    def adaptive_threshold_generate(self, logits, texts):
    # 找出一个batch中识别对的有标签数据的预测置信度的均值为伪标签筛选阈值
        max_probs, max_idx = logits.log_softmax(dim=-1).max(dim=-1)
        exp_max_probs = max_probs.exp()
        pred, non_pad_mask, length = convert_pred_to_pseudo_parallel(
                    max_idx, self.converter)
        eos_idx = self.converter.dict['[EOS]']
        correct_sample_probs = []
        for i in range(max_idx.size(0)):
            gt_idx = texts[i][:texts[i].tolist().index(eos_idx)]
            pred_idx = max_idx[i][:max_idx[i].tolist().index(eos_idx)]
            if len(gt_idx) == len(pred_idx):
                if gt_idx.tolist() == pred_idx.tolist():
                    max_probs[i][non_pad_mask[i] == False] = 0
                    exp_max_probs[i][non_pad_mask[i] == False] = 0
                    sample_prob = max_probs[i].sum(dim=-1).exp() # 累乘
                    correct_sample_probs.append(sample_prob)
        if correct_sample_probs and (sum(correct_sample_probs)/len(correct_sample_probs) >= self.opt.confident_threshold):
            threshold = sum(correct_sample_probs)/len(correct_sample_probs) # 均值
        else:
            threshold = self.opt.confident_threshold
        return threshold

    def forward(self,
                unl_img,
                aug_img,
                online,
                iteration,
                total_iter,
                T=1.0,
                l_local_feat=None,
                l_logit=None,
                l_text=None):
        loss_SemiSL = None
        l_da = None
        l_confident_ratio = 0

        self._update_ema_variables(online, iteration, self.opt.ema_alpha)
        self.target_model.eval()
        with torch.no_grad():
            unl_logit = self.target_model(unl_img,
                                          text=self.text_for_pred,
                                          is_train=False)
        self.target_model.train()
        _, unl_len, nclass = unl_logit.size()

        sequence_score, preds_index = unl_logit.log_softmax(dim=-1).max(dim=-1)
        if self.opt.use_ada_threshold:
            confident_threshold = self.adaptive_threshold_generate(l_logit, l_text)
            l_confident_threshold = confident_threshold
        else:
            confident_threshold = self.opt.confident_threshold
            l_confident_threshold = self.opt.l_confident_threshold
        if self.opt.Prediction == 'Attn':
            targets_u, non_pad_mask, unl_length = convert_pred_to_pseudo_parallel(
                preds_index, self.converter)
            sequence_score[non_pad_mask == False] = 0
            sample_prob = sequence_score.sum(dim=-1).exp()  # B
            mask = sample_prob.ge(confident_threshold) # B
            confident_mask = mask.view(-1, 1).repeat(1, unl_len)  # B T
            final_mask = (non_pad_mask & confident_mask)
        else:
            targets_u, non_pad_mask ,_= convert_pred_to_pseudo_parallel_ctc(
                preds_index, self.converter)
            sequence_score[non_pad_mask == False] = 0
            sample_prob = sequence_score.sum(dim=-1).exp()

            mask = sample_prob.ge(self.opt.confident_threshold)
            confident_mask = mask.view(-1, 1).repeat(1, unl_len)
            final_mask = (non_pad_mask & confident_mask)
            
        confident_ratio = mask.to(torch.float).mean() # 改变conf计算方式时，这里改变mask，对应的sample_prob也要改

        if iteration % 100 == 0:
            print('score', sample_prob[:2])
            print('targets_u', targets_u[:2])
            print('threshold',confident_threshold)
        if confident_ratio > 0:
            if self.opt.Prediction == 'Attn':
                unl_logit2, unl_local_feats = online(aug_img,
                                                    targets_u[:, :-1],
                                                    use_project=True,
                                                    return_local_feat=True)

                unl_score, unl_index = unl_logit2.log_softmax(dim=-1).max(dim=-1)
                unl_pred, non_pad_mask, s_unl_length = convert_pred_to_pseudo_parallel(
                    unl_index, self.converter)
                unl_pred = unl_pred[:, 1:]
                unl_score[non_pad_mask == False] = 0
                unl_prob = unl_score.sum(dim=-1).exp()
                unl_mask = unl_prob.ge(confident_threshold)

                l_score, l_index = l_logit.log_softmax(dim=-1).max(dim=-1)
                l_pred, non_pad_mask, l_length = convert_pred_to_pseudo_parallel(
                    l_index, self.converter)
                l_pred = l_pred[:, 1:]
                l_score[non_pad_mask == False] = 0
                l_prob = l_score.sum(dim=-1).exp()
                l_mask = l_prob.ge(l_confident_threshold)
                # l_mask = l_prob.ge(self.opt.l_confident_threshold)
                l_confident_ratio = l_mask.to(torch.float).mean()
                s_confident_ratio = unl_mask.to(torch.float).mean()

                if l_confident_ratio > 0 and s_confident_ratio > 0:
                    l_da = coral_loss(l_local_feat[l_mask],
                                    l_pred[l_mask],
                                    unl_local_feats[unl_mask],
                                    unl_pred[unl_mask],
                                    BLANK=self.converter.dict[' '])
                    if l_da is not None:
                        l_da = l_da * self.opt.lambda_mmd
                uda_softmax_temp = self.opt.uda_softmax_temp if self.opt.uda_softmax_temp > 0 else 1
                unl_logit1 = (unl_logit.detach() /
                            uda_softmax_temp).softmax(dim=-1)
                unl_logit2 = F.log_softmax(unl_logit2, dim=-1)
                loss_SemiSL = confident_ratio * self.kldiv_criterion(
                    unl_logit2[final_mask], unl_logit1[final_mask])
                loss_SemiSL = self.opt.lambda_cons * loss_SemiSL
            else:
                unl_logit2, unl_local_feats = online(aug_img,
                                                    targets_u,
                                                    use_project=True,
                                                    return_local_feat=True)
                unl_score, unl_index = unl_logit2.log_softmax(dim=-1).max(dim=-1)
                unl_pred, non_pad_mask, _ = convert_pred_to_pseudo_parallel_ctc(
                    unl_index, self.converter)
                # unl_pred = unl_pred[:, 1:]
                unl_score[non_pad_mask == False] = 0
                unl_prob = unl_score.sum(dim=-1).exp()
                unl_mask = unl_prob.ge(self.opt.confident_threshold)

                l_score, l_index = l_logit.log_softmax(dim=-1).max(dim=-1)
                l_pred, non_pad_mask,_ = convert_pred_to_pseudo_parallel_ctc(
                    l_index, self.converter)
                # l_pred = l_pred[:, 1:]
                l_score[non_pad_mask == False] = 0
                l_prob = l_score.sum(dim=-1).exp()
                l_mask = l_prob.ge(self.opt.l_confident_threshold)

                l_confident_ratio = l_mask.to(torch.float).mean()
                s_confident_ratio = unl_mask.to(torch.float).mean()
                if l_confident_ratio > 0 and s_confident_ratio > 0:
                    l_da = coral_loss(l_local_feat[l_mask],
                                    l_pred[l_mask],
                                    unl_local_feats[unl_mask],
                                    unl_pred[unl_mask],
                                    BLANK=self.converter.dict[' '])
                    if l_da is not None:
                        l_da = l_da * self.opt.lambda_mmd
                uda_softmax_temp = self.opt.uda_softmax_temp if self.opt.uda_softmax_temp > 0 else 1
                unl_logit1 = (unl_logit.detach() /
                            uda_softmax_temp).softmax(dim=-1)
                unl_logit2 = F.log_softmax(unl_logit2, dim=-1)
                loss_SemiSL = confident_ratio * self.kldiv_criterion(
                    unl_logit2[final_mask], unl_logit1[final_mask])
                loss_SemiSL = self.opt.lambda_cons * loss_SemiSL


        return loss_SemiSL, confident_ratio, l_da, l_confident_ratio, sample_prob

class PseudoLabel(nn.Module):

    def __init__(self, opt, online_for_init_target, converter):
        super(PseudoLabel, self).__init__()

        self.opt = opt
        self.converter = converter
        self.target_model = Model(opt)  
        self.target_model = torch.nn.DataParallel(self.target_model).to(device)
        self.target_model.train()
        if opt.Prediction == 'Attn':
            self.text_for_pred = torch.LongTensor(opt.batchSize).fill_(
                opt.sos_token_index).to(device)
            self.criterion = torch.nn.CrossEntropyLoss(
                    ignore_index=self.converter.dict['[PAD]']).to(device)
        else:
            self.text_for_pred = torch.LongTensor(opt.batchSize).fill_(0).to(device)
            self.criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    
    def forward(self,
                unl_img,
                model
                ):
        loss_SemiSL = None

        model.eval()
        with torch.no_grad():
            unl_logit = model(unl_img,
                            text=self.text_for_pred,
                            is_train=False)
        model.train()
        _, unl_len, nclass = unl_logit.size()

        unl_score, unl_index = unl_logit.log_softmax(dim=-1).max(dim=-1)
        if self.opt.Prediction == 'Attn':
            unl_pred, non_pad_mask, unl_length = convert_pred_to_pseudo_parallel(
                unl_index, self.converter)
            unl_score[non_pad_mask == False] = 0
            unl_prob = unl_score.sum(dim=-1).exp()

            mask = unl_prob.ge(self.opt.confident_threshold)
            confident_mask = mask.view(-1, 1).repeat(1, unl_len)
            unl_mask = (non_pad_mask & confident_mask)
            select_img = unl_img[mask, :, :]
            select_index = unl_pred[mask, :]
        else:
            unl_pred, non_pad_mask, unl_pred_lengths = convert_pred_to_pseudo_parallel_ctc(
                unl_index, self.converter)
            unl_score[non_pad_mask == False] = 0
            unl_prob = unl_score.sum(dim=-1).exp()

            mask = unl_prob.ge(self.opt.confident_threshold)
            confident_mask = mask.view(-1, 1).repeat(1, unl_len)
            unl_mask = (non_pad_mask & confident_mask)
            select_img = unl_img[mask, :, :]
            select_index = unl_pred[mask, :]
            select_pred_lengths = unl_pred_lengths[mask]
        
        confident_ratio = mask.to(torch.float).mean()
        if confident_ratio > 0:           
            if self.opt.Prediction == 'Attn':
                select_unl_preds = model(select_img, select_index[:, :-1])
                select_unl_target = select_index[:, 1:]
                loss_SemiSL = self.criterion(select_unl_preds.view(-1, select_unl_preds.shape[-1]),
                                select_unl_target.contiguous().view(-1))
            else:
                select_unl_preds = model(select_img, select_index)
                select_unl_preds_size = torch.IntTensor([select_unl_preds.size(1)] * select_index.size(0))
                _, select_unl_preds_index = select_unl_preds.max(2)
                select_unl_preds = select_unl_preds.log_softmax(2).permute(1,0,2)
                # select_unl_preds_str = self.converter.decode(select_unl_preds_index.data, select_unl_preds_size.data)
                loss_SemiSL = self.criterion(select_unl_preds, select_index, select_unl_preds_size, select_pred_lengths)

        return loss_SemiSL, confident_ratio, unl_prob
    
class PseudoLabelLoss(nn.Module):
    def __init__(self, opt, converter, criterion):
        super(PseudoLabelLoss, self).__init__()

        self.opt = opt
        self.converter = converter
        self.criterion = criterion

        self.PseudoLabel_prediction_model = Model(opt)
        self.PseudoLabel_prediction_model = torch.nn.DataParallel(
            self.PseudoLabel_prediction_model
        ).to(device)
        print(
            f"### loading pretrained model for PseudoLabel from {opt.model_for_PseudoLabel}"
        )
        self.PseudoLabel_prediction_model.load_state_dict(
            torch.load(opt.model_for_PseudoLabel)
        )
        self.PseudoLabel_prediction_model.eval()

    def forward(self, image_unlabel, model):

        with torch.no_grad():
            if "CTC" in self.opt.Prediction:
                PseudoLabel_pred = self.PseudoLabel_prediction_model(image_unlabel)
            else:
                idx_for_pred = (
                    torch.LongTensor(image_unlabel.size(0))
                    .fill_(self.opt.sos_token_index)
                    .to(device)
                )
                PseudoLabel_pred = self.PseudoLabel_prediction_model(
                    image_unlabel, text=idx_for_pred, is_train=False
                )

        _, PseudoLabel_index = PseudoLabel_pred.max(2)
        length_for_decode = torch.IntTensor(
            [PseudoLabel_pred.size(1)] * PseudoLabel_pred.size(0)
        )
        PseudoLabel_tmp = self.converter.decode(PseudoLabel_index, length_for_decode)
        PseudoLabel = []
        image_unlabel_unbind = torch.unbind(image_unlabel, dim=0)
        image_unlabel_filtered = []
        for image_ul, pseudo_label in zip(image_unlabel_unbind, PseudoLabel_tmp):
            # filtering unlabeled images whose prediction containing [PAD], [UNK], or [SOS] token.
            if (
                "[PAD]" in pseudo_label
                or "[UNK]" in pseudo_label
                or "[SOS]" in pseudo_label
            ):
                continue
            else:
                if "Attn" in self.opt.Prediction:
                    index_EOS = pseudo_label.find("[EOS]")
                    pseudo_label = pseudo_label[:index_EOS]

                PseudoLabel.append(pseudo_label)
                image_unlabel_filtered.append(image_ul)

        image_unlabel_filtered = torch.stack(image_unlabel_filtered, dim=0)
        Pseudo_index, Pseudo_length = self.converter.encode(
            PseudoLabel, batch_max_length=self.opt.batch_max_length
        )

        if "CTC" in self.opt.Prediction:
            preds_PL = model(image_unlabel_filtered)
            preds_PL_size = torch.IntTensor([preds_PL.size(1)] * preds_PL.size(0))
            preds_PL_log_softmax = preds_PL.log_softmax(2).permute(1, 0, 2)
            loss_SemiSL = self.criterion(
                preds_PL_log_softmax, Pseudo_index, preds_PL_size, Pseudo_length
            )
        else:
            preds_PL = model(
                image_unlabel_filtered, Pseudo_index[:, :-1]
            )  # align with Attention.forward
            target = Pseudo_index[:, 1:]  # without [SOS] Symbol
            loss_SemiSL = self.criterion(
                preds_PL.view(-1, preds_PL.shape[-1]), target.contiguous().view(-1)
            )

        return loss_SemiSL
