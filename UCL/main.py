"""
-*- coding: utf-8 -*-
@File  : main.py
@author: Yaobaochen
@Time  : 2023/8/16 下午9:09
"""

import os
import sys
import glob
import time
from torch import nn
import torch.nn.functional as F
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch_scatter import scatter

from openpoints.utils import EasyConfig
from openpoints.dataset.data_util import voxelize
from openpoints.dataset import get_features_by_keys
from openpoints.loss import build_criterion_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious

from model.base_seg import BaseSeg
from utils.random_seed import seed_everything
from dataset.build_dataloader import build_dataloader
from utils.ckpt_util import save_checkpoint, load_checkpoint
from utils.weak_util import update_ema_variables, feature_align_loss, reliable_contrastive_loss, \
    update_prototypes, unreliable_contrastive_loss, prototype_separation_loss
from utils.data_perturbation import data_perturbation


def load_data(data_path, cfg):
    label, feat = None, None
    if 's3dis' in cfg.dataset.common.NAME.lower():
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
        feat = np.clip(feat / 255., 0, 1).astype(np.float32)

    coord -= coord.min(0)

    idx_points = []
    voxel_idx, reverse_idx_part, reverse_idx_sort = None, None, None
    voxel_size = cfg.dataset.common.get('voxel_size', None)

    if voxel_size is not None:
        # idx_sort: original point indicies sorted by voxel NO.
        # voxel_idx: Voxel NO. for the sorted points
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        if cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_part = idx_sort[idx_select]
            npoints_subcloud = voxel_idx.max() + 1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle]  # idx_part: randomly sampled points of a voxel
            reverse_idx_part = np.argsort(idx_shuffle, axis=0)  # revevers idx_part to sorted
            idx_points.append(idx_part)
            reverse_idx_sort = np.argsort(idx_sort, axis=0)
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    else:
        idx_points.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort


@torch.no_grad()
def test(model, test_data_list, cfg):
    """using a part of original point cloud as input to save memory.
    Args:
        model (_type_): _description_
        test_loader (_type_): _description_
        cfg (_type_): _description_
        num_votes (int, optional): _description_. Defaults to 1.
    Returns:
        _type_: _description_
    """
    model.eval()  # set model to eval mode
    all_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    seed_everything(0)
    cfg.visualize = cfg.get('visualize', False)
    if cfg.visualize:
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    # data
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.datatransforms)

    dataset_name = cfg.dataset.common.NAME.lower()
    len_data = len(test_data_list)

    cfg.save_path = cfg.log_path + '/result'
    os.makedirs(cfg.save_path, exist_ok=True)

    gravity_dim = cfg.datatransforms.kwargs.gravity_dim

    for cloud_idx, data_path in enumerate(test_data_list):

        cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
        all_logits = []
        coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx = load_data(data_path, cfg)
        if label is not None:
            label = torch.from_numpy(label.astype(np.int).squeeze()).cuda(non_blocking=True)

        len_part = len(idx_points)
        pbar = tqdm(range(len(idx_points)))

        for idx_subcloud in pbar:
            pbar.set_description(f"Test on {cloud_idx + 1}-th cloud [{idx_subcloud}]/[{len_part}]]")

            idx_part = idx_points[idx_subcloud]
            coord_part = coord[idx_part]
            coord_part -= coord_part.min(0)

            feat_part = feat[idx_part] if feat is not None else None
            data = {'pos': coord_part}
            if feat_part is not None:
                data['x'] = feat_part
            if pipe_transform is not None:
                data = pipe_transform(data)
            if 'heights' in cfg.feature_keys and 'heights' not in data.keys():
                data['heights'] = torch.from_numpy(
                    coord_part[:, gravity_dim:gravity_dim + 1].astype(np.float32)).unsqueeze(0)
            if not cfg.dataset.common.get('variable', False):
                if 'x' in data.keys():
                    data['x'] = data['x'].unsqueeze(0)
                data['pos'] = data['pos'].unsqueeze(0)
            else:
                data['o'] = torch.IntTensor([len(coord)])
                data['batch'] = torch.LongTensor([0] * len(coord))

            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)
            data['x'] = get_features_by_keys(data, cfg.feature_keys)

            logits, __ = model(data)
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=0)
        # if not cfg.dataset.common.get('variable', False):
        #     all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)

        # average merge overlapped multi voxels logits to original point set
        idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
        all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')

        pred = all_logits.argmax(dim=1)
        if label is not None:
            cm.update(pred, label)

        if cfg.visualize:
            gt = label.cpu().numpy().squeeze() if label is not None else None
            pred = pred.cpu().numpy().squeeze()
            gt = cfg.cmap[gt, :] if gt is not None else None
            pred = cfg.cmap[pred, :]
            # output pred labels
            if 's3dis' in dataset_name:
                file_name = f'{dataset_name}-Area{cfg.dataset.common.test_area}-{cloud_idx}'
            else:
                file_name = f'{dataset_name}-{cloud_idx}'

            write_obj(coord, feat,
                      os.path.join(cfg.vis_dir, f'input-{file_name}.obj'))
            # output ground truth labels
            if gt is not None:
                write_obj(coord, gt,
                          os.path.join(cfg.vis_dir, f'gt-{file_name}.obj'))
            # output pred labels
            write_obj(coord, pred,
                      os.path.join(cfg.vis_dir, f'pred-{file_name}.obj'))

        if 'scannet' in cfg.dataset.common.NAME.lower():
            pred = pred.cpu().numpy().squeeze()
            label_int_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14,
                                 13: 16, 14: 24, 15: 28, 16: 33, 17: 34, 18: 36, 19: 39}
            pred = np.vectorize(label_int_mapping.get)(pred)
            save_file_name = data_path.split('/')[-1].split('_')
            save_file_name = save_file_name[0] + '_' + save_file_name[1] + '.txt'
            save_file_name = os.path.join(cfg.log_path+'/result/'+save_file_name)
            np.savetxt(save_file_name, pred, fmt="%d")

        if label is not None:
            tp, union, count = cm.tp, cm.union, cm.count
            miou, macc, oa, ious, accs = get_mious(tp, union, count)
            logging.info(
                f'[{cloud_idx + 1}/{len_data}] cloud,  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}'
            )
            all_cm.value += cm.value

    if label is not None:
        tp, union, count = all_cm.tp, all_cm.union, all_cm.count
        miou, macc, oa, ious, accs = get_mious(tp, union, count)
        return miou, macc, oa, ious, accs, cm
    else:
        return None, None, None, None, None, None


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y'].squeeze(-1)
        data['x'] = get_features_by_keys(data, cfg.feature_keys)

        with torch.no_grad():
            logits, __ = model(data)

        if 'mask' not in cfg.criterion_args.NAME or cfg.get('use_maks', False):
            cm.update(logits.argmax(dim=1), target)
        else:
            mask = data['mask'].bool()
            cm.update(logits.argmax(dim=1)[mask], target[mask])

    tp, union, count = cm.tp, cm.union, cm.count
    miou, macc, oa, ious, accs = get_mious(tp, union, count)

    return miou, macc, oa


def train_one_epoch(model_s, model_t, train_loader, criterion, optimizer, scheduler, epoch, cfg):

    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)

    model_s.train()  # set model to training mode
    model_t.train()

    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    Softmax = nn.Softmax(dim=1)

    loss_fa, loss_reliable, loss_unreliable, loss_ps = 0.0, 0.0, 0.0, 0.0
    for i, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1

        data['x'] = get_features_by_keys(data, cfg.feature_keys)

        mask = torch.squeeze(data['mask'])

        target = data['y'].squeeze(-1)
        target = target.flatten()
        mask = mask.flatten()
        idx = mask == 1

        data_perturb = data_perturbation(data, cfg)

        logit_s, feat_s = model_s(data_perturb)
        logit_f, feat_f = model_s(data_perturb, perturb=True)

        if cfg.ignore_index is not None:
            idx_not_ignore = (target != cfg.ignore_index)
            idx = idx & idx_not_ignore

        loss_sup = criterion(logit_s[idx], target[idx])

        with torch.no_grad():
            logit_t, feat_t = model_t(data)

        logit_s = Softmax(logit_s)
        logit_t = Softmax(logit_t)

        prob_s, label_s = logit_s.max(dim=1)
        prob_t, label_t = logit_t.max(dim=1)

        # -------------------------------- Uncertainty --------------------------------
        with torch.no_grad():
            if epoch + i == 1:
                px_s = torch.ones((cfg.dataset.train.voxel_max * cfg.batch_size, cfg.num_classes)).cuda()
                px_t = torch.ones((cfg.dataset.train.voxel_max * cfg.batch_size, cfg.num_classes)).cuda()
            else:
                px_s = Softmax(F.cosine_similarity(feat_s.unsqueeze(1), cfg.prototype.unsqueeze(0), dim=-1))
                px_t = Softmax(F.cosine_similarity(feat_t.unsqueeze(1), cfg.prototype.unsqueeze(0), dim=-1))

            conf_s = (1 / prob_s) * -torch.sum(torch.mul(torch.mul(px_s, logit_s), torch.log(logit_s)), dim=1)
            conf_t = (1 / prob_t) * -torch.sum(torch.mul(torch.mul(px_t, logit_t), torch.log(logit_t)), dim=1)
            _, idx_all = (conf_t + conf_s).sort(descending=False)

            del conf_s, conf_t, px_s, px_t

            idx_h = torch.ones((feat_s.shape[0])).cuda() == 0
            idx_h[idx_all[0:int(20 / 100 * cfg.dataset.train.voxel_max * cfg.batch_size)]] = True
            idx_h = idx_h & (label_s == label_t)

            idx_m = torch.ones((feat_s.shape[0])).cuda() == 0
            idx_m[idx_all[0:int(80 / 100 * cfg.dataset.train.voxel_max * cfg.batch_size)]] = True
            idx_m = idx_m & (label_s == label_t)

            cfg.prototype, cfg.is_prototype = update_prototypes(prob_t, label_t, feat_t, idx_h, cfg)

        # -------------------------------- feature_align_loss --------------------------------
        loss_fa = feature_align_loss(feat_t, feat_s) + feature_align_loss(feat_s, feat_f)*0.1

        # -------------------------------- reliable loss -----------------------------------
        coord = data['pos'].reshape(-1, data['pos'].shape[2])
        batch_num = torch.linspace(0, cfg.batch_size-1, steps=cfg.batch_size).to(torch.int8).unsqueeze(1).repeat(1, cfg.dataset.train.voxel_max).flatten().cuda()
        loss_reliable = reliable_contrastive_loss(label_s[idx_h], feat_s[idx_h], label_t[idx_h], feat_t[idx_h], coord[idx_h], batch_num[idx_h], cfg)

        # -------------------------------- unreliable loss ----------------------------------
        loss_unreliable = unreliable_contrastive_loss(logit_s[~idx_m], feat_s[~idx_m], cfg)

        # -------------------------------- prototype separation ----------------------------------
        loss_ps = prototype_separation_loss(label_s[idx_h], feat_s[idx_h], cfg)

        # -------------------------------- final loss ----------------------------------
        loss = loss_sup + loss_fa + loss_reliable * 0.2 + loss_unreliable * 0.5 + loss_ps * 0.3

        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model_s.parameters(), cfg.grad_norm_clip, norm_type=2)

            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()

            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        with torch.no_grad():
            model_t = update_ema_variables(model_t, model_s, 0.99, epoch)

        # update confusion matrix
        cm.update(logit_s.argmax(dim=1), target)
        loss_meter.update(loss_sup.item())

        pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}]")

    miou, macc, oa, ious, accs = cm.all_metrics()

    return loss_meter.avg, miou, macc, oa


def main(cfg):
    model_s = BaseSeg(cfg.model).cuda()
    model_t = BaseSeg(cfg.model).cuda()

    optimizer = build_optimizer_from_cfg(model_s, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    if cfg.pretraining:
        load_checkpoint(model_s, pretrained_path=cfg.model_path)
        load_checkpoint(model_t, pretrained_path=cfg.model_path)

    # build dataset
    val_loader = build_dataloader(cfg, 'val')
    train_loader = build_dataloader(cfg, 'train')

    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()

    cfg.prototype = torch.zeros((cfg.num_classes, 32)).cuda()
    cfg.is_prototype = torch.zeros((cfg.num_classes, 1), dtype=torch.int8)

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                               Start train and val
    # ---------------------------------------------------------------------------------------------------------------- #
    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Start train and val >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    best_val, macc_when_best, oa_when_best, best_epoch = 0., 0., 0., 0
    train_loss, train_miou = 0.0, 0.0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        train_loss, train_miou, train_macc, train_oa = \
            train_one_epoch(model_s, model_t, train_loader, criterion, optimizer, scheduler, epoch, cfg)

        is_best = False
        if cfg.min_val <= epoch and epoch % cfg.val_freq == 0:
            val_miou, val_macc, val_oa = validate(model_s, val_loader, cfg)
            if val_miou > best_val:
                is_best = True
                best_val = val_miou
                macc_when_best = val_macc
                oa_when_best = val_oa
                best_epoch = epoch
            save_checkpoint(cfg, model_s, epoch, optimizer, scheduler, additioanl_dict={'best_val': best_val}, is_best=is_best)

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} 'f'train_loss {train_loss:.2f}, train_miou {train_miou:.2f}'
                     f', best val miou {best_val:.2f}')

        if is_best:
            logging.info(
                f'Epoch {epoch} Find best ckpt, val_miou {best_val:.2f} '
                f'val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}'
            )

        if cfg.sched_on_epoch:
            scheduler.step(epoch)

    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Val End! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logging.info(
        f'Best val @epoch{best_epoch} , val_miou {best_val:.2f}, val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}')

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                 Start testing
    # ---------------------------------------------------------------------------------------------------------------- #

    if cfg.test_mode == 1:
        logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Start testing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        with np.printoptions(precision=2, suppress=True):
            logging.info(
                f'Test model @epoch{best_epoch}, loading the ckpt......')

        model = BaseSeg(cfg.model)
        model.cuda()

        load_checkpoint(model, pretrained_path=os.path.join(cfg.log_path, 'best.pth'))

        test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, cfg.test_data_list, cfg)

        logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Result on area 5 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        with np.printoptions(precision=2, suppress=True):
            logging.info(f'Test result @epoch{best_epoch}: test_oa {test_oa:.2f}, '
                         f'test_macc {test_macc:.2f}, test_miou {test_miou:.2f}')
            logging.info(f'iou per: {test_ious}')

        logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished all !!! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


if __name__ == "__main__":
    # load config
    config = EasyConfig()
    config.load("cfg_s3dis.yaml", recursive=True)

    # config.seed = np.random.randint(1, 10000)
    seed_everything(config.seed)

    # create log dir
    config.log_path = './log/' + config.dataset.common.NAME + '-seed_' + str(config.seed) + '-' + time.strftime(
        '%Y.%m.%d-%H:%M:%S')
    os.makedirs(config.log_path)
    if 's3dis' in config.dataset.common.NAME.lower():
        os.system('cp %s %s' % ("cfg_s3dis.yaml", config.log_path))

    # create logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s",
        handlers=[
            logging.FileHandler('%s/%s.log' % (
                config.log_path,
                config.dataset.common.NAME + '-seed' + str(config.seed) + '-' + time.strftime('%Y%m%d-%H%M'))),
            logging.StreamHandler(sys.stdout)
        ]
    )

    if config.test_mode:
        if 's3dis' in config.dataset.common.NAME.lower():
            raw_root = os.path.join(config.dataset.common.data_root, config.dataset.common.NAME, 'raw')
            data_list = sorted(os.listdir(raw_root))
            config.test_data_list = [os.path.join(raw_root, item) for item in data_list if 'Area_{}'.format(config.dataset.common.test_area) in item]

    main(config)
