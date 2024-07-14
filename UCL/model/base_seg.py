"""
Author: Yao
"""
import copy
import torch
import torch.nn as nn
from typing import List
import numpy as np

from model.layers import create_convblock1d
from model.pointnext import PointNextEncoder, PointNextDecoder
from model.pointmetabase import PointMetaBaseEncoder


def feature_perturb(data, num):
    degree = 0.1
    if num == 0:
        data = nn.Dropout(degree)(data)
        return data
    elif num == 1:
        a, b, c = data.shape
        data = data.transpose(1, 2).reshape(-1, data.shape[1])
        idx = torch.randperm(data.shape[0])[0:int(data.shape[0]*degree)]
        data[idx] = 0.0
        data = data.reshape(a, c, b).transpose(1, 2).contiguous()
        return data
    elif num == 2:
        a, b, c = data.shape
        data = data.transpose(1, 2).reshape(-1, data.shape[1])
        data = data.transpose(0, 1)
        idx = torch.randperm(data.shape[0])[0:int(data.shape[0]*degree)]
        data[idx] = 0.0
        data = data.transpose(0, 1)
        data = data.reshape(a, c, b).transpose(1, 2).contiguous()
        return data


class BaseSeg(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        encoder_args = copy.deepcopy(model_args.encoder_args)
        decoder_args = copy.deepcopy(model_args.decoder_args)
        cls_args = copy.deepcopy(model_args.cls_args)

        if encoder_args['NAME'] == 'PointNextEncoder':
            self.encoder = PointNextEncoder(encoder_args)
        if encoder_args['NAME'] == 'PointMetaBaseEncoder':
            self.encoder = PointMetaBaseEncoder(encoder_args)

        if decoder_args is not None:
            decoder_args_merged_with_encoder = copy.deepcopy(model_args.encoder_args)
            decoder_args_merged_with_encoder.NAME = encoder_args.NAME
            decoder_args_merged_with_encoder.encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder,
                                                                                                         'channel_list') else None
            self.decoder = PointNextDecoder(decoder_args_merged_with_encoder)
        else:
            self.decoder = None

        if cls_args is not None:
            if hasattr(self.decoder, 'out_channels'):
                in_channels = self.decoder.out_channels
            elif hasattr(self.encoder, 'out_channels'):
                in_channels = self.encoder.out_channels
            else:
                in_channels = cls_args.get('in_channels', None)
            cls_args.in_channels = in_channels
            self.head = SegHead(cls_args)
        else:
            self.head = None

    def forward(self, data, perturb=False):

        p, f = self.encoder.forward_seg_feat(data)

        # if perturb:
        #     f[-1] = feature_perturb(f[-1], np.random.randint(0, 3))
        #     f[-2] = feature_perturb(f[-2], np.random.randint(0, 3))
        #     f[-3] = feature_perturb(f[-3], np.random.randint(0, 3))

        if self.decoder is not None:
            f = self.decoder(p, f).squeeze(-1)

        prob = self.head(f)

        prob = prob.transpose(1, 2).reshape(-1, prob.shape[1])
        feat = f.transpose(1, 2).reshape(-1, f.shape[1])

        return prob, feat


class SegHead(nn.Module):
    def __init__(self, cls_args):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
            global_feat: global features to concat. [max,avg]. Set to None if do not concat any.
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        mlps = None
        norm_args = cls_args.norm_args
        act_args = {'act': 'relu'}
        dropout = 0.5
        global_feat = None
        num_classes = cls_args.num_classes
        in_channels = cls_args.in_channels

        if global_feat is not None:
            self.global_feat = global_feat.split(',')
            multiplier = len(self.global_feat) + 1
        else:
            self.global_feat = None
            multiplier = 1
        in_channels *= multiplier

        if mlps is None:
            mlps = [in_channels, in_channels] + [num_classes]
        else:
            if not isinstance(mlps, List):
                mlps = [mlps]
            mlps = [in_channels] + mlps + [num_classes]
        heads = []
        for i in range(len(mlps) - 2):
            heads.append(create_convblock1d(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))

        heads.append(create_convblock1d(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points):
        if self.global_feat is not None:
            global_feats = []
            for feat_type in self.global_feat:
                if 'max' in feat_type:
                    global_feats.append(torch.max(end_points, dim=-1, keepdim=True)[0])
                elif feat_type in ['avg', 'mean']:
                    global_feats.append(torch.mean(end_points, dim=-1, keepdim=True))
            global_feats = torch.cat(global_feats, dim=1).expand(-1, -1, end_points.shape[-1])
            end_points = torch.cat((end_points, global_feats), dim=1)
        logits = self.head(end_points)
        return logits
