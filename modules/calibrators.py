''' 
Adapted from https://github.com/uncbiag/LTS/blob/main/code/calibration_models.py
'''

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# LTS
class TemperatureModelLogits(nn.Module):
    def __init__(self, num_classes):
        super(TemperatureModelLogits, self).__init__()
        self.temperature_level_2_conv1 = nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv2 = nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv3 = nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv4 = nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param1 = nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param2 = nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param3 = nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv_img = nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param_img = nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2, bias=True)
        self.num_classes = num_classes
        
    def weights_init(self):
        torch.nn.init.zeros_(self.temperature_level_2_conv1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.bias.data)

    def forward(self, logits):
        temperature_1 = self.temperature_level_2_conv1(logits)
        temperature_1 += (torch.ones(1)).to(device)
        temperature_2 = self.temperature_level_2_conv2(logits)
        temperature_2 += (torch.ones(1)).to(device)
        temperature_3 = self.temperature_level_2_conv3(logits)
        temperature_3 += (torch.ones(1)).to(device)
        temperature_4 = self.temperature_level_2_conv4(logits)
        temperature_4 += (torch.ones(1)).to(device)
        temperature_param_1 = self.temperature_level_2_param1(logits)
        temperature_param_2 = self.temperature_level_2_param2(logits)
        temperature_param_3 = self.temperature_level_2_param3(logits)
        temp_level_11 = temperature_1 * torch.sigmoid(temperature_param_1) + temperature_2 * (1.0 - torch.sigmoid(temperature_param_1))
        temp_level_12 = temperature_3 * torch.sigmoid(temperature_param_2) + temperature_4 * (1.0 - torch.sigmoid(temperature_param_2))
        temp_1 = temp_level_11 * torch.sigmoid(temperature_param_3) + temp_level_12 * (1.0 - torch.sigmoid(temperature_param_3))
        temp_2 = self.temperature_level_2_conv_img(logits) + torch.ones(1).to(device)
        temp_param = self.temperature_level_2_param_img(logits)
        temperature = temp_1 * torch.sigmoid(temp_param) + temp_2 * (1.0 - torch.sigmoid(temp_param))
        sigma = 1e-8
        temperature = F.relu(temperature + torch.ones(1).to(device)) + sigma
        # 这里原代码是像素级的校准，我本来就是拉成的（b, t*c, h, w）这里怎么去取这个T有点犯难，暂时取mean
        # temperature = temperature.repeat(1, self.num_classes, 1, 1)
        temperature = temperature.mean()
        # return logits / temperature
        return temperature

class TemperatureModelLogits_v2(nn.Module):
    def __init__(self, num_classes):
        super(TemperatureModelLogits_v2, self).__init__()
        self.temperature_level_2_conv = nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param = nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv_img = nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param_img = nn.Conv2d(num_classes, 1, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2, bias=True)
        self.num_classes = num_classes

    def forward(self, logits):
        temperature = self.temperature_level_2_conv(logits) + torch.ones(1).to(device)
        temperature_param = self.temperature_level_2_param(logits)
        temp_img = self.temperature_level_2_conv_img(logits) + torch.ones(1).to(device)
        temp_param_img = self.temperature_level_2_param_img(logits)
        
        temp_level = temperature * torch.sigmoid(temperature_param) + temp_img * (1.0 - torch.sigmoid(temperature_param))
        temperature = temp_level * torch.sigmoid(temp_param_img)
        
        sigma = 1e-8
        temperature = torch.relu(temperature + torch.ones(1).to(device)) + sigma
        temperature = temperature.mean()
        
        return temperature

# TS
class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, converter):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.converter = converter
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, input, label, text_for_pred):
        logits = self.model(input, text_for_pred)
        text_for_loss, length_for_loss = self.converter.encode(label, batch_max_length=25)
        logits = logits[:, :text_for_loss.shape[1] - 1, :]
        ts_preds = self.temperature_scale(logits)
        return ts_preds

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        #print(logits.size())
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1),logits.size(2))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, mode = 'attn'):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.to(device)
        if mode == 'attn':
            criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
        else:
            criterion = nn.CTCLoss(zero_infinity=True).to(device)
        ece_criterion = _ECELoss().to(device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        text_list = []
        length_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                batch_size = input.size(0)
                input = input.to(device)
                #print(input.shape)
                text_for_pred = torch.LongTensor(batch_size).fill_(
                self.converter.dict['[SOS]']).to(device)
                text_for_loss, length_for_loss = self.converter.encode(label, batch_max_length=25)
                target = text_for_loss[:, 1:]
                #print(input.shape(), label.shape())
                logits = self.model(input, text_for_pred, is_train=False)
                if mode == 'attn':
                    # logits = logits[:, :text_for_loss.shape[1] - 1, :]
                    logits_list.append(logits)
                else:
                    logits_list.append(logits)
                    text_list.append(text_for_loss)
                    length_list.append(length_for_loss)
                labels_list.append(target)
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)
            if mode == 'ctc':
                text_list = torch.cat(text_list).to(device)
                length_list = torch.cat(length_list).to(device)

        # Calculate NLL and ECE before temperature scaling
        # criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        before_temperature_nll = criterion(logits.contiguous().view(-1, logits.shape[-1]), labels.contiguous().view(-1)).item()
        before_temperature_ece = ece_criterion(logits.contiguous().view(-1, logits.shape[-1]), labels.contiguous().view(-1)).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        #print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=25)

        def eval():
            optimizer.zero_grad()
            if mode == 'attn':
                loss = criterion(self.temperature_scale(logits).contiguous().view(-1, logits.shape[-1]), labels.contiguous().view(-1))
            else:
                preds_size = torch.IntTensor([logits.size(1)] * logits.size(0))
                loss = criterion(self.temperature_scale(logits).log_softmax(2).permute(1, 0, 2), text_list, preds_size, length_list)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = criterion(self.temperature_scale(logits).contiguous().view(-1, logits.shape[-1]), labels.contiguous().view(-1)).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits).contiguous().view(-1, logits.shape[-1]), labels.contiguous().view(-1)).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        
        return self.temperature.item()

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece