"""
Author: PointNeXt
"""
import numpy as np
import torch
from easydict import EasyDict as edict

from openpoints.transforms import build_transforms_from_cfg
from utils.random_seed import seed_everything
from dataset.s3dis_dataset import s3dis_dataset


def collate_fn_offset_train(batches):
    """collate fn and offset
    """
    pts, feats, labels, heights, mask = [], [], [], [], []
    for i in range(0, len(batches)):
        pts.append(batches[i]['pos'])
        feats.append(batches[i]['x'])
        labels.append(batches[i]['y'])
        heights.append(batches[i]['heights'])
        mask.append(batches[i]['mask'])

    data = {'pos': torch.stack(pts, dim=0),
            'x': torch.stack(feats, dim=0),
            'y': torch.stack(labels, dim=0),
            'heights': torch.stack(heights, dim=0),
            'mask': torch.stack(mask, dim=0),
            }
    return data


def collate_fn_offset_val(batches):
    """collate fn and offset
    """
    pts, feats, labels, heights = [], [], [], []
    for i in range(0, len(batches)):
        pts.append(batches[i]['pos'])
        feats.append(batches[i]['x'])
        labels.append(batches[i]['y'])
        heights.append(batches[i]['heights'])

    data = {'pos': torch.unsqueeze(torch.cat(pts, dim=0), dim=0),
            'x': torch.unsqueeze(torch.cat(feats, dim=0), dim=0),
            'y': torch.unsqueeze(torch.cat(labels, dim=0), dim=0),
            'heights': torch.unsqueeze(torch.cat(heights, dim=0), dim=0),
            }
    return data


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, split):
    if split == 'train':
        batch_size = cfg.batch_size
    else:
        batch_size = cfg.val_batch_size

    dataset_cfg = cfg.dataset
    datatransforms_cfg = cfg.datatransforms

    if datatransforms_cfg is not None:
        # in case only val or test transforms are provided.
        if split not in datatransforms_cfg.keys() and split in ['val', 'test']:
            trans_split = 'val'
        else:
            trans_split = split
        data_transform = build_transforms_from_cfg(trans_split, datatransforms_cfg)
    else:
        data_transform = None

    if split not in dataset_cfg.keys() and split in ['val', 'test']:
        dataset_split = 'test' if split == 'val' else 'val'
    else:
        dataset_split = split
    split_cfg = dataset_cfg.get(dataset_split, edict())
    if split_cfg.get('split', None) is None:  # add 'split' in dataset_split_cfg
        split_cfg.split = split
    split_cfg.transform = data_transform

    dataset = s3dis_dataset(cfg, dataset_cfg.common, split_cfg)

    if split == 'train':
        collate_fn_offset = collate_fn_offset_train
    else:
        collate_fn_offset = collate_fn_offset_val

    shuffle = split == 'train'
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=cfg.num_workers,
                                             worker_init_fn=lambda k: seed_everything(cfg.seed + (k * 10000)),
                                             drop_last=split == 'train',
                                             shuffle=shuffle,
                                             collate_fn=collate_fn_offset,
                                             pin_memory=True)
    return dataloader
