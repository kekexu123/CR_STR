import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import editdistance as ed


class LabelSmoothingLoss(nn.Module):
    def __init__(self,
                 ignore_index: int,
                 alpha: float,
                 normalize_length: bool = True,
                 **kwargs):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length

    def forward(self, preds: torch.Tensor, target: torch.Tensor, *args) -> torch.Tensor:
        batch_size = preds.size(0)
        preds = preds.view(-1, preds.size(-1))
        target = target.view(-1)
        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(preds)
        true_dist.fill_(self.smoothing / (preds.size(-1) - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(preds, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
    
class GraduatedLabelSmoothingAttn(nn.Module):
    def __init__(self, 
                ignore_index=0,
                alpha=0.0,
                normalize_length=True,
                **kwargs):
        super(GraduatedLabelSmoothingAttn, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length
    def forward(self, input, target, *args):    # input: tensor(B, T, C)  target: tensor(B, T) eos_token:3, pad_token:0
        self.size = input.size(-1)
        batch_size = input.size(0)
        input = input.view(-1, input.shape[-1])
        target = target.view(-1)
        
        pred_probability, _ = torch.softmax(input, dim=1).max(1)
        smoothing = self.smoothing * torch.ones_like(input)
        smoothing[pred_probability >= 0.7, :] = 3 * self.smoothing
        smoothing[pred_probability <= 0.3, :] = 0.0
        true_dist = smoothing / (self.size - 1)
        confidence = 1 - true_dist.sum(-1)
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        #target = target.masked_fill(ignore, 0)  0
        true_dist.scatter_(1, target.unsqueeze(1), confidence.unsqueeze(1))
        kl = self.criterion(torch.log_softmax(input, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
    
class ClassAwareLablesmoothing_v1(nn.Module):
    def __init__(self, 
                ignore_index=0,
                alpha=0.0,
                smooth_tail=0.0,
                cls_num_list=None,
                head_cls=None,
                normalize_length=True,
                **kwargs):
        super(ClassAwareLablesmoothing_v1, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.cross_entropy = nn.CrossEntropyLoss()
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length
        self.cls_smooth = smooth_tail + (alpha - smooth_tail) * (np.array(cls_num_list) - min(cls_num_list)) / (max(cls_num_list) - min(cls_num_list))
        self.head_cls = head_cls
    def forward(self, input, target, *args):
        self.size = input.size(-1)
        batch_size = input.size(0)
        input = input.view(-1, input.shape[-1])
        target = target.view(-1)
        index = torch.nonzero(target != self.padding_idx).squeeze()
        input = input[index, :]
        target = target[index]

        pred_probability, pred_index = torch.softmax(input, dim=1).max(1)

        smoothing = torch.zeros_like(input)
        i=0
        for j in torch.nonzero(target==3).squeeze().view(-1):
            edit_d = ed.eval(target[i:j].tolist(), pred_index[i:j].tolist())
            if edit_d != 0:
                example_smoothing = 1-(1-self.smoothing)**(1.0 / (edit_d))
            else:
                example_smoothing = 1e-12
            example_smoothing = example_smoothing * torch.ones_like(input[i:j,:])
            cls_smoothing = 1e-12 * torch.ones_like(input[i:j,:])
            for t in range(j-i):
                if target[i+t] in self.head_cls:
                    cls_smoothing[t, :] = torch.tensor(self.cls_smooth).to(input.device)
        
            smoothing[i:j, :] = example_smoothing * cls_smoothing
            i=j+1
        
        true_dist = smoothing / (self.size - 1)
        confidence = 1 - true_dist.sum(-1)
        true_dist.scatter_(1, target.unsqueeze(1), confidence.unsqueeze(1))
        kl = self.criterion(torch.log_softmax(input, dim=1), true_dist)
        denom = len(target) if self.normalize_length else batch_size
        return kl.sum() / denom
    
class ClassAwareLablesmoothing_v2(nn.Module):
    '''
    相比于v1,校准强度从×改成＋
    '''
    def __init__(self, 
                ignore_index=0,
                alpha=0.0,
                smooth_tail=0.0,
                cls_num_list=None,
                head_cls=None,
                normalize_length=True,
                **kwargs):
        super(ClassAwareLablesmoothing_v2, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.cross_entropy = nn.CrossEntropyLoss()
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length
        self.cls_smooth = smooth_tail + (alpha - smooth_tail) * (np.array(cls_num_list) - min(cls_num_list)) / (max(cls_num_list) - min(cls_num_list))
        self.head_cls = head_cls
    def forward(self, input, target, *args):
        self.size = input.size(-1)
        batch_size = input.size(0)
        input = input.view(-1, input.shape[-1])
        target = target.view(-1)
        index = torch.nonzero(target != self.padding_idx).squeeze()
        input = input[index, :]
        target = target[index]

        pred_probability, pred_index = torch.softmax(input, dim=1).max(1)

        smoothing = torch.zeros_like(input)
        i=0
        for j in torch.nonzero(target==3).squeeze().view(-1):
            edit_d = ed.eval(target[i:j].tolist(), pred_index[i:j].tolist())
            if edit_d != 0:
                example_smoothing = 1-(1-self.smoothing)**(1.0 / (edit_d))
            else:
                example_smoothing = 1e-12
            example_smoothing = example_smoothing * torch.ones_like(input[i:j,:])
            cls_smoothing = 1e-12 * torch.ones_like(input[i:j,:])
            for t in range(j-i):
                if target[i+t] in self.head_cls:
                    cls_smoothing[t, :] = torch.tensor(self.cls_smooth).to(input.device)
        
            smoothing[i:j, :] = example_smoothing + cls_smoothing
            i=j+1
        
        true_dist = smoothing / (self.size - 1)
        confidence = 1 - true_dist.sum(-1)
        true_dist.scatter_(1, target.unsqueeze(1), confidence.unsqueeze(1))
        kl = self.criterion(torch.log_softmax(input, dim=1), true_dist)
        denom = len(target) if self.normalize_length else batch_size
        return kl.sum() / denom
    
class ClassAwareLablesmoothing_v3(nn.Module):
    '''
    相比于v2, 固定了类别最大校准强度为0.1,并修改了example_smoothing
    '''
    def __init__(self, 
                ignore_index=0,
                alpha=0.0,
                smooth_tail=0.0,
                cls_num_list=None,
                head_cls=None,
                normalize_length=True,
                **kwargs):
        super(ClassAwareLablesmoothing_v3, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.cross_entropy = nn.CrossEntropyLoss()
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length
        self.cls_smooth = smooth_tail + (0.05 - smooth_tail) * (np.array(cls_num_list) - min(cls_num_list)) / (max(cls_num_list) - min(cls_num_list))
        self.head_cls = head_cls
    def forward(self, input, target, *args):
        self.size = input.size(-1)
        batch_size = input.size(0)
        input = input.view(-1, input.shape[-1])
        target = target.view(-1)
        index = torch.nonzero(target != self.padding_idx).squeeze()
        input = input[index, :]
        target = target[index]

        pred_probability, pred_index = torch.softmax(input, dim=1).max(1)

        smoothing = torch.zeros_like(input)
        i=0
        for j in torch.nonzero(target==3).squeeze().view(-1):
            edit_d = ed.eval(target[i:j].tolist(), pred_index[i:j].tolist())
            example_smoothing = 1-(1-self.smoothing)**(edit_d * self.smoothing)
            example_smoothing = example_smoothing * torch.ones_like(input[i:j,:])
            cls_smoothing = torch.zeros_like(input[i:j,:])
            for t in range(j-i):
                if target[i+t] in self.head_cls:
                    cls_smoothing[t, :] = torch.tensor(self.cls_smooth).to(input.device)
        
            smoothing[i:j, :] = example_smoothing + cls_smoothing
            i=j+1
        
        true_dist = smoothing / (self.size - 1)
        confidence = 1 - true_dist.sum(-1)
        true_dist.scatter_(1, target.unsqueeze(1), confidence.unsqueeze(1))
        kl = self.criterion(torch.log_softmax(input, dim=1), true_dist)
        denom = len(target) if self.normalize_length else batch_size
        return kl.sum() / denom
    
class ClassAwareLablesmoothing_v4(nn.Module):
    '''
    相比于v3, 修改了example_smoothing,使之与cer和self.smoothing成正相关,而不再是编辑距离
    '''
    def __init__(self, 
                ignore_index=0,
                alpha=0.0,
                smooth_tail=0.0,
                cls_num_list=None,
                head_cls=None,
                normalize_length=True,
                **kwargs):
        super(ClassAwareLablesmoothing_v4, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.cross_entropy = nn.CrossEntropyLoss()
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length
        self.cls_smooth = smooth_tail + (0.05 - smooth_tail) * (np.array(cls_num_list) - min(cls_num_list)) / (max(cls_num_list) - min(cls_num_list))
        self.head_cls = head_cls
    def forward(self, input, target, *args):
        self.size = input.size(-1)
        batch_size = input.size(0)
        input = input.view(-1, input.shape[-1])
        target = target.view(-1)
        index = torch.nonzero(target != self.padding_idx).squeeze()
        input = input[index, :]
        target = target[index]

        pred_probability, pred_index = torch.softmax(input, dim=1).max(1)

        smoothing = torch.zeros_like(input)
        i=0

        for j in torch.nonzero(target == 3).squeeze().view(-1):
            if i < j:  # 确保i和j不相等，避免除以零
                edit_d = ed.eval(target[i:j].tolist(), pred_index[i:j].tolist())
                if len(target[i:j].tolist()) > 0:  # 再次检查以确保列表不为空
                    beta = edit_d / len(target[i:j].tolist())
                    example_smoothing = 1 - (1 - beta) ** (self.smoothing)
                    example_smoothing = example_smoothing * torch.ones_like(input[i:j,:])
                    cls_smoothing = torch.zeros_like(input[i:j,:])
                    for t in range(j-i):
                        if target[i+t] in self.head_cls:
                            cls_smoothing[t, :] = torch.tensor(self.cls_smooth).to(input.device)
                
                    smoothing[i:j, :] = example_smoothing + cls_smoothing
            i = j + 1
        
        true_dist = smoothing / (self.size - 1)
        confidence = 1 - true_dist.sum(-1)
        true_dist.scatter_(1, target.unsqueeze(1), confidence.unsqueeze(1))
        kl = self.criterion(torch.log_softmax(input, dim=1), true_dist)
        denom = len(target) if self.normalize_length else batch_size
        return kl.sum() / denom
    
class LabelAwareSmoothing(nn.Module):
    def __init__(self, 
                ignore_index=0,
                cls_num_list=None,
                alpha=0.0, 
                smooth_tail=0.0, 
                shape='concave', 
                power=None):
        super(LabelAwareSmoothing, self).__init__()

        self.padding_idx = ignore_index
        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)

        if shape == 'concave':
            self.smooth = smooth_tail + (alpha - smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'linear':
            self.smooth = smooth_tail + (alpha - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)

        elif shape == 'convex':
            self.smooth = alpha + (alpha - smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (alpha - smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        if torch.cuda.is_available():
            self.smooth = self.smooth.cuda()

    def forward(self, input, target, *args):
        input = input.view(-1, input.shape[-1])
        target = target.view(-1)
        index = torch.nonzero(target != self.padding_idx).squeeze()
        input = input[index, :]
        target = target[index]

        smoothing = self.smooth[target]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(input, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss

        return loss.mean()

class ACLS(nn.Module):

    def __init__(self,
                 pos_lambda: float = 1.0,
                 neg_lambda: float = 0.1,
                 alpha: float = 0.1,    
                 margin: float = 10.0,
                 num_classes: int = 200,
                 ignore_index: int = -100):
        super().__init__()
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.alpha = alpha
        self.margin = margin
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "reg"

    def get_reg(self, inputs, targets):
        max_values, indices = inputs.max(dim=1)
        max_values = max_values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        indicator = (max_values.clone().detach() == inputs.clone().detach()).float()

        batch_size, num_classes = inputs.size()
        num_pos = batch_size * 1.0
        num_neg = batch_size * (num_classes - 1.0)

        neg_dist = max_values.clone().detach() - inputs
        
        pos_dist_margin = F.relu(max_values - self.margin)
        neg_dist_margin = F.relu(neg_dist - self.margin)

        pos = indicator * pos_dist_margin ** 2
        neg = (1.0 - indicator) * (neg_dist_margin ** 2)

        reg = self.pos_lambda * (pos.sum() / num_pos) + self.neg_lambda * (neg.sum() / num_neg)
        return reg


    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        loss_ce = self.cross_entropy(inputs, targets)

        loss_reg = self.get_reg(inputs, targets)
        loss = loss_ce + self.alpha * loss_reg

        return loss