import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import pdb

from ..fda.build_model import Deeplab
from ..losses.loss_builder import build_loss


class FDASegmentor(nn.Module):
    """
    unsupervised domain adaptation segmentor
    """
    def __init__(self, cfg):
        super(FDASegmentor, self).__init__()

        self.model = Deeplab(num_classes=19)
        self.loss = build_loss(cfg)
        self.cfg = cfg

    def forward(self, source, target=None, source_label=None, source_label_onehot=None, target_label=None, target_label_onehot=None):
        # source domain
        if not self.training or self.cfg.MODEL.DISCRIMINATOR.WEIGHT or self.cfg.DATASET.TARGET.SOURCE_LOSS_WEIGHT>0:
            s_logits = self.model(source)

        if self.training:
            # target domain
            t_logits = self.model(target)
            
            # defaut reg_weight
            if self.cfg.MODEL.DISCRIMINATOR.LAMBDA_ENTROPY_WEIGHT or self.cfg.MODEL.DISCRIMINATOR.LAMBDA_KLDREG_WEIGHT:
                reg_val_matrix = torch.ones_like(target_label).type_as(t_logits)
                reg_val_matrix[target_label==255]=0
                reg_val_matrix = reg_val_matrix.unsqueeze(dim=1)
                reg_ignore_matrix = 1 - reg_val_matrix
                reg_weight = torch.ones_like(t_logits)
                reg_weight_val = reg_weight * reg_val_matrix
                reg_weight_ignore = reg_weight * reg_ignore_matrix
                del reg_ignore_matrix, reg_weight, reg_val_matrix
            
            losses = {}
            
            # pseudo label target seg loss
            if target_label is not None:
                target_seg_loss = self.cfg.DATASET.TARGET.PSEUDO_LOSS_WEIGHT * self.loss(t_logits, target_label)
                losses.update({"target_seg_loss": target_seg_loss})

            # entropy reg
            if self.cfg.MODEL.DISCRIMINATOR.LAMBDA_ENTROPY_WEIGHT > 0:
                entropy_reg_loss = entropyloss(t_logits, reg_weight_ignore)
                entropy_reg_loss =  entropy_reg_loss * self.cfg.MODEL.DISCRIMINATOR.LAMBDA_ENTROPY_WEIGHT
                losses.update({"entropy_reg_loss": entropy_reg_loss})

            # kld reg
            if self.cfg.MODEL.DISCRIMINATOR.LAMBDA_KLDREG_WEIGHT > 0:
                kld_reg_loss = kldloss(t_logits, reg_weight_val)
                kld_reg_loss =  kld_reg_loss * self.cfg.MODEL.DISCRIMINATOR.LAMBDA_KLDREG_WEIGHT
                losses.update({"kld_reg_loss": kld_reg_loss})

            return losses
        
        return s_logits

def entropyloss(logits, weight=None):
    """
    logits:     N * C * H * W 
    weight:     N * 1 * H * W
    """
    val_num = weight[weight>0].numel()
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    num_classed = logits.size()[1]
    entropy = -torch.softmax(logits, dim=1) * weight * logits_log_softmax
    entropy_reg = torch.sum(entropy) / val_num
    return entropy_reg

def kldloss(logits, weight):
    """
    logits:     N * C * H * W 
    weight:     N * 1 * H * W
    """
    val_num = weight[weight>0].numel()
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    num_classes = logits.size()[1]
    kld = - 1/num_classes * weight * logits_log_softmax
    kld_reg = torch.sum(kld) / val_num
    return kld_reg