import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler, RandomSampler
from torch.autograd import Variable

from ..datasets.loader.dataset import BaseDataset
from ..datasets.loader.gtav_dataset import GTAVDataset
from ..datasets.loader.cityscapes_dataset import CityscapesDataset
from ..datasets.loader.bdd_dataset import BDDDataset

from ..datasets.metrics.miou import intersectionAndUnionGPU
from ..datasets.metrics.acc import acc, acc_with_hist
from ..models.losses.ranger import Ranger
from ..models.losses.cos_annealing_with_restart import CosineAnnealingLR_with_Restart
from ..models.registry import DATASET

import os
import time
import numpy as np
import tqdm
import pdb
import json
from PIL import Image

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(s_mask):
    # mask: numpy array of the mask
    mask = np.copy(s_mask.cpu())
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def label_mapping(input, mapping):
    output = np.copy(input.cpu())
    for ind in range(len(mapping)):
        output[input.cpu() == mapping[ind][0]] = mapping[ind][1]
    return torch.tensor(np.array(output, dtype=np.int64)).cuda(non_blocking=True)

def eval_net(net, cfg, gpu):

    with open("../data/info.json", 'r') as fp:
        info = json.load(fp)
    
    mapping = np.array(info['label2train'], dtype=np.int)
    
    # train dataset
    result = []
    early_stopping = cfg.TRAIN.EARLY_STOPPING
    anns = cfg.DATASET.ANNS
    image_dir = cfg.DATASET.IMAGEDIR
    use_aug = cfg.DATASET.USE_AUG
    scales = cfg.TEST.RESIZE_SIZE
    bs = cfg.TEST.BATCH_SIZE
    num_work = cfg.TEST.NUM_WORKER
    use_flip = cfg.TEST.USE_FLIP

    # init net
    dist.init_process_group(
        backend='nccl', 
        init_method='tcp://127.0.0.1:6789', 
        world_size=cfg.TRAIN.N_PROC_PER_NODE,
        rank=gpu
        )
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:{}".format(gpu))
    net.to(device)
    
    # val dataset  
    val_anns = cfg.DATASET.VAL.ANNS
    val_image_dir = cfg.DATASET.VAL.IMAGEDIR
    val = DATASET[cfg.DATASET.VAL.TYPE](val_anns, val_image_dir)
    val_sampler = DistributedSampler(val, num_replicas=cfg.TEST.N_PROC_PER_NODE, rank=gpu)
    val_data = DataLoader(val, bs, num_workers=num_work, sampler=val_sampler)

    if gpu == 0:
        print('Eval Size: {}'.format(scales))
        print('Use Flip: {}'.format(use_flip))
    
    with torch.no_grad():
        net.eval()
        n_class = cfg.MODEL.PREDICTOR.NUM_CLASSES
        intersection_sum = 0
        union_sum = 0

        save_path = '/content/drive/MyDrive/IAST_result3/result'
        save_label = '/content/drive/MyDrive/cityscapes_label'
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except Exception:
                pass

        # if not os.path.exists(save_label):
        #     try:
        #         os.makedirs(save_label)
        #     except Exception:
        #         pass

        if gpu == 0:
            pbar = tqdm.tqdm(total=len(val_data))

        for i, b in enumerate(val_data):
            if gpu == 0 :
                pbar.update(1)
            images = b[0].cuda(non_blocking=True)
            labels = b[1].type(torch.LongTensor).cuda(non_blocking=True)
            #labels = label_mapping(labels, mapping)

            ii = 0
            pred_result = []
            for scale in scales:
                tmp_images = F.interpolate(images, scale[::-1], mode='bilinear', align_corners=True)
                logits = F.softmax(net(tmp_images), dim=1)

                if use_flip:
                    flip_logits = F.softmax(net(torch.flip(tmp_images, dims=[3])), dim=1)
                    logits += torch.flip(flip_logits, dims=[3])

                logits = F.interpolate(logits, labels.size()[1:], mode='bilinear', align_corners=True)
                pred_result.append((1+ii)*logits)
                ii = ii + 0.5
            result = sum(pred_result)

            label_pred = result.max(dim=1)[1]

            intersection, union = intersectionAndUnionGPU(label_pred, labels, n_class)
            intersection_sum += intersection
            union_sum += union

            if(i%50==0):
                print(intersection/union)

            # Save ground truth
            # img = colorize_mask(labels[0])
            # name = b[2][0].split('/')[-1]
            # img.save(save_label + '/' + name)

            # img = colorize_mask(labels[1])
            # name = b[2][1].split('/')[-1]
            # img.save(save_label + '/' + name)
            
            # Save predicted label
            # img = colorize_mask(label_pred[0])
            # # img = np.copy(label_pred[0].cpu())
            # # img = Image.fromarray(img.astype(np.uint8))
            # name = b[2][0].split('/')[-1]
            # img.save(save_path + '/' + name)

            # img = colorize_mask(label_pred[1])
            # # img = np.copy(label_pred[1].cpu())
            # # img = Image.fromarray(img.astype(np.uint8))
            # name = b[2][1].split('/')[-1]
            # img.save(save_path + '/' + name)
        
        if gpu == 0:
            pbar.close()
            

        dist.all_reduce(intersection_sum), dist.all_reduce(union_sum)
        intersection_sum = intersection_sum.cpu().numpy()
        union_sum = union_sum.cpu().numpy()

        if gpu == 0:
            iu = intersection_sum / (union_sum + 1e-10)
            mean_iu = np.mean(iu)
            print('val_miou: {:.4f}'.format(mean_iu) + print_iou_list(iu))


def print_iou_list(iou_list):
    res = ''
    for i, iou in enumerate(iou_list):
        res += ', {}: {:.4f}'.format(i, iou)
    return res

