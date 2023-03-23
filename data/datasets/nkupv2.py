# encoding: utf-8
import os
import os.path as osp
from .bases import BaseImageDataset

class NKUPv2(BaseImageDataset):

    def __init__(self, root='', pid_begin = 0, **kwargs):
        super(NKUPv2, self).__init__()
        self.dataset_dir         = osp.join(root, 'NKUP2_0126', 'NKUP2')
        self.train_dir           = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir           = osp.join(self.dataset_dir, 'bounding_box_test', 'query')
        self.sameapp_gal_dir     = osp.join(self.dataset_dir, 'bounding_box_test', 'gallery_sameapp')
        self.crossapp_modgal_dir = osp.join(self.dataset_dir, 'bounding_box_test', 'gallery_crossapp_moderate')
        self.crossapp_dragal_dir = osp.join(self.dataset_dir, 'bounding_box_test', 'gallery_crossapp_dramatic')
        self._check_before_run(self.train_dir, self.query_dir, self.sameapp_gal_dir, self.crossapp_modgal_dir, self.crossapp_dragal_dir)

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
        print("  crossapp_mod_gal | {:6d}  | {:9d}  |  {:7d}  |  {:9d}".format(self.num_ca_mod_gal_pids, self.num_ca_mod_gal_imgs, self.num_ca_mod_gal_apps, self.num_ca_mod_gal_cams))
        print("  crossapp_dra_gal | {:6d}  | {:9d}  |  {:7d}  |  {:9d}".format(self.num_ca_dra_gal_pids, self.num_ca_dra_gal_imgs, self.num_ca_dra_gal_apps, self.num_ca_dra_gal_cams))
        print("  ------------------------------------------------------------------------")


    def _process_dir(self, dir_path, relabel=False):
        pid_container = set()
        dataset = []
        for _id in os.listdir(dir_path):
            pid_container.add(int(_id))
            for _app in os.listdir(osp.join(dir_path, _id)):
                for _pic in os.listdir(osp.join(dir_path, _id, _app)):
                    self.cam_set.add(int(_pic[12:14]))
        
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        camid2label = {camid: label for label, camid in enumerate(self.cam_set)}
        app_id = -1
        
        for _id in os.listdir(dir_path):
            pid = int(_id)
            if relabel: pid = pid2label[pid]
            for _app in os.listdir(osp.join(dir_path, _id)):
                app_id += 1
                for _pic in os.listdir(osp.join(dir_path, _id, _app)):
                    img_path = osp.join(dir_path, _id, _app, _pic)
                    camid = int(_pic[12:14])
                    if relabel: camid = camid2label[camid]

                    dataset.append((img_path, self.pid_begin + pid, camid, app_id))
        
        return dataset
