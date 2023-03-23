# encoding: utf-8
import os
import os.path as osp
from .bases import BaseImageDataset

class NKUPv2CM(BaseImageDataset):
    dirs = ['NKUP2', 'NKUP2_Edge', 'NKUP2_Parsing']

    def __init__(self, root='', pid_begin = 0, **kwargs):
        super(NKUPv2CM, self).__init__()
        self.train_dir           = [osp.join(root, 'NKUP2_0126', _dir, 'bounding_box_train') for _dir in self.dirs]
        self.query_dir           = [osp.join(root, 'NKUP2_0126', _dir, 'bounding_box_test', 'query') for _dir in self.dirs]
        self.sameapp_gal_dir     = [osp.join(root, 'NKUP2_0126', _dir, 'bounding_box_test', 'gallery_sameapp') for _dir in self.dirs]
        self.crossapp_modgal_dir = [osp.join(root, 'NKUP2_0126', _dir, 'bounding_box_test', 'gallery_crossapp_moderate') for _dir in self.dirs]
        self.crossapp_dragal_dir = [osp.join(root, 'NKUP2_0126', _dir, 'bounding_box_test', 'gallery_crossapp_dramatic') for _dir in self.dirs]
        self._check_before_run(*self.train_dir, *self.query_dir, *self.sameapp_gal_dir, *self.crossapp_modgal_dir, *self.crossapp_dragal_dir)

        self.pid_begin = pid_begin
        self.cam_set = set()
        self.train = self._process_dir(self.train_dir, relabel=True)
        self.query = self._process_dir(self.query_dir, relabel=False)
        self.sameapp_gallery = self._process_dir(self.sameapp_gal_dir, relabel=False)
        self.crossapp_mod_gallery = self._process_dir(self.crossapp_modgal_dir, relabel=False)
        self.crossapp_dra_gallery = self._process_dir(self.crossapp_dragal_dir, relabel=False)

        self.num_train_pids, self.num_train_imgs, self.num_train_apps, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_apps, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_sa_gal_pids, self.num_sa_gal_imgs, self.num_sa_gal_apps, self.num_sa_gal_cams = self.get_imagedata_info(self.sameapp_gallery)
        self.num_ca_mod_gal_pids, self.num_ca_mod_gal_imgs, self.num_ca_mod_gal_apps, self.num_ca_mod_gal_cams = self.get_imagedata_info(self.crossapp_mod_gallery)
        self.num_ca_dra_gal_pids, self.num_ca_dra_gal_imgs, self.num_ca_dra_gal_apps, self.num_ca_dra_gal_cams = self.get_imagedata_info(self.crossapp_dra_gallery)
        self.print_dataset_statistics()

    def print_dataset_statistics(self):
        print("  NKUP2 Dataset statistics: ")
        print("  ------------------------------------------------------------------------")
        print("  subset           |  # ids  |  # images  |  # apps  |  # cameras ")
        print("  ------------------------------------------------------------------------")
        print("  train            | {:6d}  | {:9d}  |  {:7d}  |  {:9d}".format(self.num_train_pids, self.num_train_imgs, self.num_train_apps, self.num_train_cams))
        print("  query            | {:6d}  | {:9d}  |  {:7d}  |  {:9d}".format(self.num_query_pids, self.num_query_imgs, self.num_query_apps, self.num_query_cams))
        print("  sameapp_gallery  | {:6d}  | {:9d}  |  {:7d}  |  {:9d}".format(self.num_sa_gal_pids, self.num_sa_gal_imgs, self.num_sa_gal_apps, self.num_sa_gal_cams))
        print("  cross_query      | {:6d}  | {:9d}  |  {:7d}  |  {:9d}".format(self.num_ca_mod_gal_pids, self.num_ca_mod_gal_imgs, self.num_ca_mod_gal_apps, self.num_ca_mod_gal_cams))
        print("  cross_gallery    | {:6d}  | {:9d}  |  {:7d}  |  {:9d}".format(self.num_ca_dra_gal_pids, self.num_ca_dra_gal_imgs, self.num_ca_dra_gal_apps, self.num_ca_dra_gal_cams))
        print("  ------------------------------------------------------------------------")


    def _process_dir(self, dir_paths, relabel=False):
        rgb_path = dir_paths[0]
        pid_container = set()
        dataset = []
        for _id in os.listdir(rgb_path):
            pid_container.add(int(_id))
            for _app in os.listdir(osp.join(rgb_path, _id)):
                for _pic in os.listdir(osp.join(rgb_path, _id, _app)):
                    self.cam_set.add(int(_pic[12:14]))

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        camid2label = {camid: label for label, camid in enumerate(self.cam_set)}
        app_id = -1
        
        for _id in os.listdir(rgb_path):
            pid = int(_id)
            if relabel: pid = pid2label[pid]
            for _app in os.listdir(osp.join(rgb_path, _id)):
                app_id += 1
                for _pic in os.listdir(osp.join(rgb_path, _id, _app)):

                    img_path_rgb = osp.join(dir_paths[0], _id, _app, _pic)
                    img_path_edge = osp.join(dir_paths[1], _id, _app, _pic[:-3]+'png')
                    img_path_parsing = osp.join(dir_paths[2], _id, _app, _pic[:-3]+'png')

                    camid = int(_pic[12:14])
                    if relabel: camid = camid2label[camid]
                    if not relabel: app_id = int(_app[-1])

                    dataset.append(((img_path_rgb, img_path_edge, img_path_parsing), pid, camid, app_id))
        
        return dataset
