import random
import numpy as np
import torch
import collections
from scipy.linalg import expm, norm
import copy


class PointCloudRotation(object):
    def __init__(self, angle=[0, 0, 0]):
        self.angle = np.array(angle) * np.pi

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, data):
        if hasattr(data, 'keys'):
            device = data['pos'].device
        else:
            device = data.device

        if isinstance(self.angle, collections.Iterable):
            rot_mats = []
            for axis_ind, rot_bound in enumerate(self.angle):
                theta = 0
                axis = np.zeros(3)
                axis[axis_ind] = 1
                if rot_bound is not None:
                    theta = np.random.uniform(-rot_bound, rot_bound)
                rot_mats.append(self.M(axis, theta))
            # Use random order
            np.random.shuffle(rot_mats)
            rot_mat = torch.tensor(rot_mats[0] @ rot_mats[1] @ rot_mats[2], dtype=torch.float32, device=device)
        else:
            raise ValueError()

        """ DEBUG
        from openpoints.dataset import vis_multi_points
        old_points = data.cpu().numpy()
        # old_points = data['pos'].numpy()
        # new_points = (data['pos'] @ rot_mat.T).numpy()
        new_points = (data @ rot_mat.T).cpu().numpy()
        vis_multi_points([old_points, new_points])
        End of DEBUG"""

        if hasattr(data, 'keys'):
            data['pos'] = data['pos'] @ rot_mat.T
            if 'normals' in data:
                data['normals'] = data['normals'] @ rot_mat.T
        else:
            data = data @ rot_mat.T
        return data


class PointCloudScaling(object):
    def __init__(self,
                 scale=[2. / 3, 3. / 2],
                 anisotropic=True,
                 scale_xyz=[True, True, True],
                 mirror=[0, 0, 0],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.mirror = torch.from_numpy(np.array(mirror))
        self.use_mirroring = torch.sum(torch.tensor(self.mirror)>0) != 0

    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        if self.use_mirroring:
            assert self.anisotropic==True
            self.mirror = self.mirror.to(device)
            mirror = (torch.rand(3, device=device) > self.mirror).to(torch.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        if hasattr(data, 'keys'):
            data['pos'] *= scale
        else:
            data *= scale
        return data


class RandomHorizontalFlip(object):
    def __init__(self, upright_axis, aug_prob=0.95, **kwargs):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.D = 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])
        self.aug_prob = aug_prob

    def __call__(self, data):
        if random.random() < self.aug_prob:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = torch.max(data['pos'])
                    data['pos'][:, curr_ax] = coord_max - data['pos'][:, curr_ax]
                    if 'normals' in data:
                        data['normals'][:, curr_ax] = -data['normals'][:, curr_ax]

        return data


def data_perturbation(data, cfg):
    data_perturb = copy.deepcopy(data)

    scale = PointCloudScaling([0.95, 1.05])
    data_perturb = scale(data_perturb)

    rotation = PointCloudRotation([0, 0, 2])
    data_perturb = rotation(data_perturb)

    # flip = RandomHorizontalFlip('x')
    # data_perturb = flip(data_perturb)

    return data_perturb
