import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
from math import ceil
import collections

import torch
from torch.utils.data import Dataset
from utils.data_util import crop_pc, voxelize


def generate_scene_mask(labels, weak_ratio):

    mask = np.zeros((labels.shape[0], 1), dtype=np.int8)

    counts = collections.Counter(np.squeeze(labels.astype(np.long)))

    for key, value in dict(counts).items():
        idx_label = [idx for idx, label in enumerate(labels) if label == key]

        idx_ratio = np.random.permutation(np.array(idx_label))[0:ceil(weak_ratio/100 * value)]

        mask[idx_ratio] = 1

    return mask


class s3dis_dataset(Dataset):
    classes = ['ceiling',
               'floor',
               'wall',
               'beam',
               'column',
               'window',
               'door',
               'chair',
               'table',
               'bookcase',
               'sofa',
               'board',
               'clutter']
    num_classes = 13
    num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                              650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
    class2color = {'ceiling': [0, 255, 0],
                   'floor': [0, 0, 255],
                   'wall': [0, 255, 255],
                   'beam': [255, 255, 0],
                   'column': [255, 0, 255],
                   'window': [100, 100, 255],
                   'door': [200, 200, 100],
                   'table': [170, 120, 200],
                   'chair': [255, 0, 0],
                   'sofa': [200, 100, 100],
                   'bookcase': [10, 200, 100],
                   'board': [200, 200, 200],
                   'clutter': [50, 50, 50]}
    cmap = [*class2color.values()]
    gravity_dim = 2
    """S3DIS dataset, loading the subsampled entire room as input without block/sphere subsampling.
    number of points per room in average, median, and std: (794855.5, 1005913.0147058824, 939501.4733064277)
    Args:
        data_root (str, optional): Defaults to 'data/S3DIS/s3disfull'.
        test_area (int, optional): Defaults to 5.
        voxel_size (float, optional): the voxel size for donwampling. Defaults to 0.04.
        voxel_max (_type_, optional): subsample the max number of point per point cloud. Set None to use all points.  Defaults to None.
        split (str, optional): Defaults to 'train'.
        transform (_type_, optional): Defaults to None.
        loop (int, optional): split loops for each epoch. Defaults to 1.
        presample (bool, optional): wheter to downsample each point cloud before training. Set to False to downsample on-the-fly. Defaults to False.
        variable (bool, optional): where to use the original number of points. The number of point per point cloud is variable. Defaults to False.
    """

    def __init__(self, cfg, dataset_cfg, split_cfg):

        super().__init__()

        self.voxel_size = dataset_cfg.voxel_size
        self.data_root = dataset_cfg.data_root
        self.test_area = dataset_cfg.test_area
        self.NAME = dataset_cfg.NAME

        self.split = split_cfg.split
        self.transform = split_cfg.transform
        self.voxel_max = split_cfg.voxel_max
        self.loop = split_cfg.loop
        self.presample = split_cfg.presample
        self.variable = split_cfg.variable
        self.shuffle = split_cfg.shuffle

        raw_root = os.path.join(self.data_root, self.NAME, 'raw')
        self.raw_root = raw_root

        processed_root = os.path.join(self.data_root, self.NAME)
        if self.split == 'val':
            filename = os.path.join(
                processed_root,
                f's3dis_{self.split}_area{self.test_area}_{self.voxel_size:.3f}.pkl')
        else:
            filename = os.path.join(
                processed_root,
                f's3dis_{self.split}_area{self.test_area}_{self.voxel_size:.3f}_weak_{str(cfg.weak_ratio)}.pkl')

        if not os.path.exists(filename):
            data_list = sorted(os.listdir(raw_root))
            data_list = [item[:-4] for item in data_list if 'Area_' in item]
            if self.split == 'train':
                self.data_list = [
                    item for item in data_list if not 'Area_{}'.format(self.test_area) in item]
            else:
                self.data_list = [
                    item for item in data_list if 'Area_{}'.format(self.test_area) in item]

        if self.presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            for item in tqdm(self.data_list,
                             desc=f'Loading S3DISFull {self.split} split on Test Area {self.test_area}'):
                per_data = []
                data_path = os.path.join(raw_root, item + '.npy')
                cdata = np.load(data_path).astype(np.float32)
                cdata[:, :3] -= np.min(cdata[:, :3], 0)
                if self.voxel_size:
                    coord, feat, label = cdata[:, 0:3], cdata[:, 3:6], cdata[:, 6:7]
                    uniq_idx = voxelize(coord, self.voxel_size)
                    coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
                    per_data.append(coord)
                    per_data.append(feat)
                    per_data.append(label.astype(np.int8))

                if self.split == 'train':
                    mask = generate_scene_mask(label, cfg.weak_ratio)
                    per_data.append(mask)

                self.data.append(per_data)

            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif self.presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")

        self.data_idx = np.arange(len(self.data))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {self.split} set")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        if self.presample:
            if self.split == 'val':
                coord, feat, label = self.data[data_idx][0], self.data[data_idx][1], self.data[data_idx][2]
            elif self.split == 'train':
                coord, feat, label, mask = self.data[data_idx][0], self.data[data_idx][1], self.data[data_idx][2], \
                                           self.data[data_idx][3]

                coord, feat, label, mask = crop_pc(
                    coord, feat, label, mask, self.split, self.voxel_size, self.voxel_max,
                    downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)
        else:
            data_path = os.path.join(
                self.raw_root, self.data_list[data_idx] + '.npy')
            cdata = np.load(data_path).astype(np.float32)
            cdata[:, :3] -= np.min(cdata[:, :3], 0)
            coord, feat, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:7]
            coord, feat, label = crop_pc(
                coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)

        # TODO: do we need to -np.min in cropped data?

        label = label.squeeze(-1).astype(np.long)
        data = {'pos': coord, 'x': feat, 'y': label}
        # pre-process.
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' not in data.keys():
            data['heights'] = torch.from_numpy(coord[:, self.gravity_dim:self.gravity_dim + 1].astype(np.float32))

        if self.split == 'train':
            data['mask'] = torch.from_numpy(mask)

        return data

    def __len__(self):
        return len(self.data_idx) * self.loop
        # return 1   # debug


"""debug 
from openpoints.dataset import vis_multi_points
import copy
old_data = copy.deepcopy(data)
if self.transform is not None:
    data = self.transform(data)
vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3].numpy()], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3].numpy()])
"""
