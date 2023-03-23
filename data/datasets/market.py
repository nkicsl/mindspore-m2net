# encoding: utf-8

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class Market(BaseImageDataset):
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(Market, self).__init__()
        self.train_dir   = osp.join(root, 'market', 'rgb', 'bounding_box_train')
        self.query_dir   = osp.join(root, 'market', 'rgb', 'query')
        self.gallery_dir = osp.join(root, 'market', 'rgb', 'bounding_box_test')

        self.train = self._process_dir(self.train_dir, relabel=True)
        self.query = self._process_dir(self.query_dir, relabel=False)
        self.sameapp_gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(self.train, self.query, self.sameapp_gallery)

        self.num_train_pids,   self.num_train_imgs,   self.num_train_apps,   self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids,   self.num_query_imgs,   self.num_query_apps,   self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_apps, self.num_gallery_cams = self.get_imagedata_info(self.sameapp_gallery)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid, pid))

        return dataset
