"""
-*- coding: utf-8 -*-
@File  : weak_util.py
@author: Yaobaochen
@Time  : 2023/8/16
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    return ema_model


def update_prototypes(prob_t, label_t, feat_t, idx_all, cfg):

    prototype, is_prototype = cfg.prototype, cfg.is_prototype

    for i in range(cfg.num_classes):
        idx_cls = (label_t == i) & idx_all
        if idx_cls.sum() > 0:
            prob_t_cls = prob_t[idx_cls]
            feat_t_cls = feat_t[idx_cls]

            max_sort_t = prob_t_cls.sort(descending=True)[1]

            num = 1000

            if max_sort_t.shape[0] > num:
                feat_cls = feat_t_cls[max_sort_t[:num]]
            else:
                feat_cls = feat_t_cls

            if is_prototype[i] == 0:
                prototype[i] = torch.mean(feat_cls, dim=0).unsqueeze(0)
                is_prototype[i] = 1
            else:
                prototype[i] = 0.75 * prototype[i] + 0.25 * torch.mean(feat_cls, dim=0).unsqueeze(0)

    # prototype = torch.where(torch.isnan(prototype), torch.full_like(prototype, 0), prototype)

    return prototype, is_prototype


def feature_align_loss(feat, feat_t):

    PairwiseDistance = nn.PairwiseDistance(p=2)
    loss = torch.mean(PairwiseDistance(feat, feat_t))

    return loss


def reliable_contrastive_loss(label_s, feat_s, label_t, feat_t, coord, batch_num, cfg):

    temp = 5
    eps = 1e-10

    with torch.no_grad():
        idx_choice = np.random.choice(range(feat_t.shape[0]), size=3000)
        neg = feat_t[idx_choice]
        neg_label = label_t[idx_choice]

        # 距离与相似度权重计算
        pos_batch_label = batch_num
        neg_batch_label = batch_num[idx_choice]

        pos_pesu = pos_batch_label.unsqueeze(1).repeat(1, neg.shape[0])
        neg_pesu = neg_batch_label.unsqueeze(1).repeat(1, feat_t.shape[0])
        batch_mask = torch.ones((feat_t.shape[0], neg.shape[0]), dtype=torch.int8).cuda()
        batch_mask *= (pos_pesu == neg_pesu.T)

        dist = torch.cdist(coord, coord[idx_choice])
        d_weight = (1 - (dist / torch.max(dist, dim=1)[0].unsqueeze(1).repeat(1, neg.shape[0]))) * 5
        d_weight *= batch_mask

        # dist = torch.cdist(pos, neg)  k_dist = dist.topk(k=dist.shape[0], dim=1, largest=False)
        sim = F.cosine_similarity(neg.unsqueeze(1), cfg.prototype.unsqueeze(0), dim=-1)
        c_weight = torch.gather(sim, dim=1, index=label_s.unsqueeze(1).repeat(1, neg.shape[0]).T).T * 5

        weight = c_weight + d_weight

        label_pos = torch.tensor(label_s, dtype=torch.int8)
        label_neg = torch.tensor(neg_label, dtype=torch.int8)

        pos_pesu = label_pos.unsqueeze(1).repeat(1, neg.shape[0])
        neg_pesu = label_neg.unsqueeze(1).repeat(1, feat_t.shape[0])
        neg_mask = torch.ones((feat_t.shape[0], neg.shape[0]), dtype=torch.int8).cuda()
        neg_mask *= (pos_pesu != neg_pesu.T)
        pos_mask = (neg_mask == 0)

    del dist, d_weight, sim, c_weight, batch_mask, pos_pesu, neg_pesu
    torch.cuda.empty_cache()

    neg = (feat_s @ neg.T) / temp

    up = (torch.exp(neg) * pos_mask).mean(-1)
    down = torch.exp(neg) * neg_mask
    down = (down * weight).sum(-1)

    loss = torch.mean(-torch.log(torch.clip(up / torch.clip(up + down, eps), eps)))

    if torch.isnan(loss):
        loss = 0

    return loss


def unreliable_contrastive_loss(logit_s, feat_s, cfg):
    temp = 0.5
    eps = 1e-10

    # compute cosine similarity
    x = feat_s.unsqueeze(1)
    y = cfg.prototype.unsqueeze(0)
    output = F.cosine_similarity(x, y, dim=-1)

    with torch.no_grad():

        neg_mask = (output < 0.1) | (logit_s < 0.1)

    # up = (torch.exp(output / temp) * pos_mask).sum(-1)

    down = (torch.exp(output / temp) * neg_mask).sum(-1)

    loss = -torch.mean(torch.log(torch.clip(1 / torch.clip(1 + down, eps), eps)))

    return loss


def prototype_separation_loss(label_s, feat_s, cfg):

    temp = 0.5
    eps = 1e-10

    prototype = torch.zeros((cfg.num_classes, 32)).cuda()
    for c in range(cfg.num_classes):
        idx_cls = (label_s == c)
        if idx_cls.sum() > 0:
            prototype[c] = torch.mean(feat_s[idx_cls], dim=0).unsqueeze(0)

    x = prototype.unsqueeze(1)
    y = cfg.prototype.unsqueeze(0)
    output = F.cosine_similarity(x, y, dim=-1)

    pos_mask = torch.eye(cfg.num_classes).cuda()

    up = (torch.exp(output / temp) * pos_mask).sum(-1)
    down = (torch.exp(output / temp) * (pos_mask == 0)).sum(-1)

    loss = -torch.mean(torch.log(torch.clip(up / torch.clip(1 + down, eps), eps)))

    return loss
