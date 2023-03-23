# encoding: utf-8

import argparse
import os
import sys
import torch
import random
import numpy as np
from torch.backends import cudnn

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

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total: {total_num/1024/1024}M, Trainable:{trainable_num/1024/1024}M')

def run_train():
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

    dataset, num_classes = create_dataset(
            cfg.DATASETS.ROOT_DIR,
            ims_per_id=cfg.DATALOADER.NUM_INSTANCE,
            ids_per_batch=cfg.IDS_PER_BATCH,
            mean=cfg.INPUT.PIXEL_MEAN,
            std=cfg.INPUT.PIXEL_STD,
            resize_h_w=cfg.INPUT.IMG_SIZE,
            padding=cfg.INPUT.PADDING,
            dataset_name=cfg.DATASETS.NAMES,
            dataset_type= cfg.MODEL.CROSSMODEITY,
    )
    batch_num = dataset.get_dataset_size()
    network = build_model_mindspore(cfg, num_classes, num_apps, True)

    # pre_trained
    if config.pre_trained:
        print('Load model from', cfg.MODEL.PRETRAIN_PATH)
        load_param_into_net(network, load_checkpoint(cfg.MODEL.PRETRAIN_PATH))

    reid_loss = M2NetLoss(cfg, num_classes, num_apps)
    optimizer = network.get_optimizer(cfg, reid_loss, batch_num)

    timestamp = time.strftime("%Y%m%d_%H%M%S") + '_' + str(config.rank)

    logfile = SavingTimeMonitor.open_file(
        config.train_log_path if config.rank_save_ckpt_flag else None,
        timestamp=timestamp,
    )

    time_cb = SavingTimeMonitor(data_size=batch_num, logfile=logfile)

    callbacks = [time_cb]
    if cfg.SOLVER.LOG_PERIOD is None:
            cfg.SOLVER.LOG_PERIOD = batch_num


    loss_cb = SavingLossMonitor(
        per_print_times=cfg.SOLVER.LOG_PERIOD,
        logfile=logfile,
        init_info=str(cfg),
    )
    callbacks.append(loss_cb)
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=int(cfg.SOLVER.CHECKPOINT_PERIOD * batch_num),
        keep_checkpoint_max=10,
    )
    save_ckpt_path = os.path.join(cfg.OUTPUT_DIR, timestamp + '-ckpt' + '/')
    ckpt_cb = ModelCheckpoint(
        config=ckpt_config,
        directory=save_ckpt_path,
        prefix='{}'.format(0),
    )
    callbacks.append(ckpt_cb)

    network_loss = NetworkWithCell(network, reid_loss)
    network_loss = TrainOneStepCell(network_loss, optimizer, center_loss_weight=cfg.SOLVER.CENTER_LOSS_WEIGHT)
    model = Model(network_loss)

    copyfile(join("./tools/train.py"), join(output_dir, "train.py"))
    copyfile(join("./modeling/baseline.py"), join(output_dir, "baseline.py"))

    model.train(cfg.SOLVER.MAX_EPOCHS, dataset, callbacks=callbacks, dataset_sink_mode=False)
    
    
if __name__ == '__main__':
    run_train()
    
