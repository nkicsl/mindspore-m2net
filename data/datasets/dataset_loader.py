# encoding: utf-8


import os.path as osp
from PIL import Image
import numpy as np

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset():
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, appid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        img = np.asarray(img)

        return img, pid, appid, camid, img_path


class CMImageDataset():
    """Image Person Cross Modality ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (img_rgb_path, img_edge_path, img_parsing_path), pid, camid, appid = self.dataset[index]

        img_rgb = read_image(img_rgb_path)
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
        img_rgb = np.asarray(img_rgb)

        img_edge = read_image(img_edge_path)
        if self.transform is not None:
            img_edge = self.transform(img_edge)
        img_edge = np.asarray(img_edge)

        img_parsing = read_image(img_parsing_path)
        if self.transform is not None:
            img_parsing = self.transform(img_parsing)
        img_parsing = np.asarray(img_parsing)

        return img_rgb, img_edge, img_parsing, pid, appid, camid, img_rgb_path
