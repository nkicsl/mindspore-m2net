# encoding: utf-8

from importlib.resources import path
import logging
from .datasets import init_dataset, ImageDataset, CMImageDataset
from .triplet_sampler import ReIDDistributedSampler
import mindspore.dataset as ds
import mindspore.dataset.vision as C
import math
import random


def random_erasing(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
    """ Random erasing augmentation

    Args:
        img: input image
        probability: augmentation probability
        sl: min erasing area
        sh: max erasing area
        r1: erasing ratio
        mean: erasing color
    Returns:
        augmented image
    """
    if random.uniform(0, 1) > probability:
        return img

    ch, height, width = img.shape

    for _ in range(100):
        area = height * width

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < width and h < height:
            x1 = random.randint(0, height - h)
            y1 = random.randint(0, width - w)
            if ch == 3:
                img[0, x1:x1 + h, y1:y1 + w] = mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = mean[2]
            else:
                img[0, x1:x1 + h, y1:y1 + w] = mean[0]
            return img

    return img

def create_dataset(
        image_folder,
        ims_per_id=4,
        ids_per_batch=32,
        batch_size=None,
        resize_h_w=(384, 128),
        padding=10,
        num_parallel_workers=8,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        data_part='train',
        dataset_name='market1501',
        dataset_type=False,
):
    """ Crate dataloader for ReID

    Args:
        image_folder: path to image folder
        ims_per_id: number of ids in batch
        ids_per_batch: number of imager per id
        batch_size: batch size (if None then batch_size=ims_per_id*ids_per_batch)
        resize_h_w: height and width of image
        padding: size of augmentation padding
        num_parallel_workers: number of parallel workers
        mean: image mean value for normalization
        std: image std value for normalization
        data_part: part of data: train|test|query
        dataset_name: name of dataset: market1501|dukemtmc

    Returns:
        if train data_part:
            dataset
        else:
            dataset, camera_ids, person_ids
    """
    mean = [m * 255 for m in mean]
    std = [s * 255 for s in std]

    if batch_size is None:
        batch_size = ids_per_batch * ims_per_id

    full_dataset = init_dataset(dataset_name, image_folder)
    num_classes = full_dataset.num_train_pids

    if data_part == 'train':
        subset = full_dataset.train
    elif data_part == 'query':
        subset = full_dataset.query
    elif data_part == 'eval_same':
        subset = full_dataset.sameapp_gallery
    elif data_part == 'eval_cross':
        subset = full_dataset.crossapp_mod_gallery + full_dataset.crossapp_dra_gallery
    elif data_part == 'eval_cross_mod':
        subset = full_dataset.crossapp_mod_gallery
    elif data_part == 'eval_cross_dra':
        subset = full_dataset.crossapp_dra_gallery
    else:
        raise ValueError(f'Unknown data_part {data_part}')

    if dataset_type == False:
        reid_dataset = ImageDataset(subset)
    else: reid_dataset = CMImageDataset(subset)

    sampler, shuffle = None, None

    _, pids, camids, appids = list(zip(*subset))

    if data_part == 'train':

        sampler = ReIDDistributedSampler(
            subset,
            batch_id=ids_per_batch,
            batch_image=ims_per_id,
        )

        transforms_list = [
            C.Resize(resize_h_w),
            C.RandomHorizontalFlip(),
            C.Pad(padding),
            C.RandomCrop(resize_h_w),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
        ]
    else:
        shuffle = False

        transforms_list = [
            C.Resize(resize_h_w),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
        ]

    dataset = ds.GeneratorDataset(
        source=reid_dataset,
        column_names=['image', 'label'],
        sampler=sampler,
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list,
        input_columns=["image"],
        num_parallel_workers=num_parallel_workers,
    )

    if data_part == 'train':
        dataset = dataset.map(
            operations=random_erasing,
            input_columns=["image"],
            num_parallel_workers=num_parallel_workers,
        )

    dataset = dataset.batch(batch_size, drop_remainder=False)

    if data_part == 'train':
        return dataset, num_classes

    return dataset, num_classes, camids, pids, appids
