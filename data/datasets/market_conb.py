# encoding: utf-8

import os
import os.path as osp
from .bases import BaseImageDataset


class MarketCM(BaseImageDataset):
    dirs = ['rgb', 'edge', 'parsing']

    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(MarketCM, self).__init__()
        self.train_dir           = [osp.join(root, 'market', _dir, 'bounding_box_train') for _dir in self.dirs]
        self.query_dir           = [osp.join(root, 'market', _dir, 'query') for _dir in self.dirs]
        self.sameapp_gal_dir     = [osp.join(root, 'market', _dir, 'bounding_box_test') for _dir in self.dirs]

        self.train = self._process_dir(self.train_dir, relabel=True)
        self.query = self._process_dir(self.query_dir, relabel=False)
        self.sameapp_gallery = self._process_dir(self.sameapp_gal_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(self.train, self.query, self.sameapp_gallery)

        self.num_train_pids,   self.num_train_imgs,   self.num_train_apps,   self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids,   self.num_query_imgs,   self.num_query_apps,   self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_apps, self.num_gallery_cams = self.get_imagedata_info(self.sameapp_gallery)

    def _process_dir(self, dir_path, relabel=False):
        rgb_path = dir_path[0]
        pid_container = set()
        dataset = []

        for _pic in os.listdir(osp.join(rgb_path)):
            if _pic[0] == '-': continue  # junk images are just ignored
            pid = int(_pic[:4])
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for _pic in os.listdir(osp.join(rgb_path)):
            if _pic[0] == '-': continue  # junk images are just ignored
            pid = int(_pic[:4])
            camid = int(_pic[6:7])
            appid = pid
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            img_path_rgb = osp.join(dir_path[0], _pic)
            img_path_edge = osp.join(dir_path[1], _pic[:-3]+'png')
            img_path_parsing = osp.join(dir_path[2], _pic[:-3]+'png')

            dataset.append( ( (img_path_rgb, img_path_edge, img_path_parsing), pid, camid, appid ) )

        return dataset
