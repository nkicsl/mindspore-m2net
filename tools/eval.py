# encoding: utf-8

import argparse
import os
import sys
import random
import numpy as np

sys.path.append('.')
from config import cfg
from data import make_data_loader
from data import create_dataset
from modeling import build_model, build_model_mindspore
from utils.lr_scheduler import WarmupMultiStepLR
from utils.logger import setup_logger
from tools.train import do_train
from tools.test import do_test
from shutil import copyfile
from os.path import join
from mindspore.communication.management import init, get_rank, get_group_size
from utils.callbacks import SavingLossMonitor, SavingTimeMonitor
from torchstat import stat
from modeling.m2net import M2NetLoss
from modeling.cell_wrapper import TrainOneStepCell, NetworkWithCell
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.train.model import Model
from utils.reid_metric import compute_metrics

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total: {total_num/1024/1024}M, Trainable:{trainable_num/1024/1024}M')

def extract_feature(model, dataset):
    """ Extract dataset features from model """
    data_loader = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)

    features = []

    for data in data_loader:
        images_ = data["image"]

        images = Tensor.from_numpy(images_)
        outputs = model(images)
        ff = outputs

        fnorm = mnp.sqrt((ff ** 2).sum(axis=1, keepdims=True))
        ff = ff / fnorm.expand_as(ff)

        features.append(ff.asnumpy())

    return np.concatenate(features, axis=0)

def run_eval():
    parser = argparse.ArgumentParser(description="NKUP Re-ID Baseline")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.MODEL.CROSSMODEITY: assert "cm" in cfg.DATASETS.NAMES, "Not match for he cross modity and dataset"
    else: assert "cm" not in cfg.DATASETS.NAMES , "Not match for he cross modity and dataset"

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("nkup_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True


    logger.info("Evaluate Only")
    q_dataset, num_classes, q_camids, q_pids, _ = create_dataset(
        cfg.DATASETS.ROOT_DIR,
        ims_per_id=cfg.DATALOADER.NUM_INSTANCE,
        ids_per_batch=cfg.IDS_PER_BATCH,
        mean=cfg.INPUT.PIXEL_MEAN,
        std=cfg.INPUT.PIXEL_STD,
        resize_h_w=cfg.INPUT.IMG_SIZE,
        padding=cfg.INPUT.PADDING,
        data_part='query',
        dataset_name=cfg.DATASETS.NAMES,
        dataset_type= cfg.MODEL.CROSSMODEITY,
    )
    same_dataset, _, s_camids, s_pids, _ = create_dataset(
        cfg.DATASETS.ROOT_DIR,
        ims_per_id=cfg.DATALOADER.NUM_INSTANCE,
        ids_per_batch=cfg.IDS_PER_BATCH,
        mean=cfg.INPUT.PIXEL_MEAN,
        std=cfg.INPUT.PIXEL_STD,
        resize_h_w=cfg.INPUT.IMG_SIZE,
        padding=cfg.INPUT.PADDING,
        data_part='query',
        dataset_name=cfg.DATASETS.NAMES,
        dataset_type= cfg.MODEL.CROSSMODEITY,
    )
    cross_dataset, _, c_camids, c_pids, _ = create_dataset(
        cfg.DATASETS.ROOT_DIR,
        ims_per_id=cfg.DATALOADER.NUM_INSTANCE,
        ids_per_batch=cfg.IDS_PER_BATCH,
        mean=cfg.INPUT.PIXEL_MEAN,
        std=cfg.INPUT.PIXEL_STD,
        resize_h_w=cfg.INPUT.IMG_SIZE,
        padding=cfg.INPUT.PADDING,
        data_part='query',
        dataset_name=cfg.DATASETS.NAMES,
        dataset_type= cfg.MODEL.CROSSMODEITY,
    )
    mod_dataset, _, m_camids, m_pids, _ = create_dataset(
        cfg.DATASETS.ROOT_DIR,
        ims_per_id=cfg.DATALOADER.NUM_INSTANCE,
        ids_per_batch=cfg.IDS_PER_BATCH,
        mean=cfg.INPUT.PIXEL_MEAN,
        std=cfg.INPUT.PIXEL_STD,
        resize_h_w=cfg.INPUT.IMG_SIZE,
        padding=cfg.INPUT.PADDING,
        data_part='query',
        dataset_name=cfg.DATASETS.NAMES,
        dataset_type= cfg.MODEL.CROSSMODEITY,
    )
    dra_dataset, _, d_camids, d_pids, _ = create_dataset(
        cfg.DATASETS.ROOT_DIR,
        ims_per_id=cfg.DATALOADER.NUM_INSTANCE,
        ids_per_batch=cfg.IDS_PER_BATCH,
        mean=cfg.INPUT.PIXEL_MEAN,
        std=cfg.INPUT.PIXEL_STD,
        resize_h_w=cfg.INPUT.IMG_SIZE,
        padding=cfg.INPUT.PADDING,
        data_part='query',
        dataset_name=cfg.DATASETS.NAMES,
        dataset_type= cfg.MODEL.CROSSMODEITY,
    )

    network = build_model_mindspore(cfg, num_classes, num_apps, False)

    if os.path.isdir(cfg.TEST.WEIGHT):
        weights = []
        _file_prefix = ''
        for _file in os.listdir(cfg.TEST.WEIGHT):
            if _file[-2:] == 'pt':
                num = _file.split('_')[-1].split('.')[0]
                _file_prefix = _file.split(num)[0]
                weights.append(int(num))
        assert len(weights) > 0, 'Not find weight'
        weights.sort()
        weights = [os.path.join(cfg.TEST.WEIGHT, _file_prefix+str(num)+'.pt') for num in weights]
    else:
        weights = [cfg.TEST.WEIGHT]

    for i, weight_path in enumerate(weights):
        print('Load model from', weight_path)
        ret, _ = load_param_into_net(network, load_checkpoint(weight_path))
    
    qf = extract_feature(network, q_dataset)
    same_f = extract_feature(network, same_dataset)
    cross_f = extract_feature(network, cross_dataset)
    mod_f = extract_feature(network, mod_dataset)
    dra_f = extract_feature(network, dra_dataset)

    feats = np.concatenate([qf, same_f])
    pids = np.concatenate([q_ids, s_ids])
    cam_ids = np.concatenate([q_cams, s_cams])
    r, m_ap = compute_metrics(feats, pids, cam_ids, len(q_ids))
    print(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'.format(
                m_ap,
                r[0], r[2], r[4], r[9],
            )
        )

    feats = np.concatenate([qf, cross_f])
    pids = np.concatenate([q_ids, c_ids])
    cam_ids = np.concatenate([q_cams, c_cams])
    r, m_ap = compute_metrics(feats, pids, cam_ids, len(q_ids))
    print(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'.format(
                m_ap,
                r[0], r[2], r[4], r[9],
            )
        )

    feats = np.concatenate([qf, mod_f])
    pids = np.concatenate([q_ids, m_ids])
    cam_ids = np.concatenate([q_cams, m_cams])
    r, m_ap = compute_metrics(feats, pids, cam_ids, len(q_ids))
    print(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'.format(
                m_ap,
                r[0], r[2], r[4], r[9],
            )
        )

    feats = np.concatenate([qf, dra_f])
    pids = np.concatenate([q_ids, d_ids])
    cam_ids = np.concatenate([q_cams, d_cams])
    r, m_ap = compute_metrics(feats, pids, cam_ids, len(q_ids))
    print(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'.format(
                m_ap,
                r[0], r[2], r[4], r[9],
            )
        )
   
if __name__ == '__main__':
    run_eval)
    
