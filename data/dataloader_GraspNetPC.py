import os
import glob

import h5py
import numpy as np
import open3d as o3d
from augmentation import density, drop_hole, p_scan
import torch.utils.data as data

# from grasp_datautils import jitter_pointcloud, scale_to_unit_cube, rotate_pc, random_rotate_one_axis, jitter_pointcloud_adaptive
# from utils.pc_utils_Norm import farthest_point_sample_no_curv_np

import pdb


class GraspNetPointClouds(data.Dataset):
    def __init__(self, dataroot, partition='train'):
        super(GraspNetPointClouds).__init__()
        self.partition = partition

    def __getitem__(self, item):
        o3d_pointcloud = o3d.io.read_point_cloud(self.pc_list[item])
        pointcloud = np.asarray(o3d_pointcloud.points)                  # 1024, 3

        pointcloud = pointcloud.astype(np.float32)
        path = self.pc_list[item].split('.x')[0]
        label = np.copy(self.label[item])

        if self.partition == 'train':
            pointcloud_aug = pointcloud
            if np.random.random() > 0.5:
                pointcloud_aug = density(pointcloud_aug)
            if np.random.random() > 0.5:
                pointcloud_aug = drop_hole(pointcloud_aug)
            if np.random.random() > 0.5:
                pointcloud_aug = p_scan(pointcloud_aug)
        else:
            pointcloud_aug = pointcloud

        data_item = {}
        data_item['PC'] = pointcloud
        data_item['Label'] = label
        data_item['PC_Aug'] = pointcloud_aug

        return (item, pointcloud, label, pointcloud_aug)

    def __len__(self):
        return len(self.pc_list)

    # def get_data_loader(self, batch_size, num_workers, drop_last, shuffle=True):
    #     return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=drop_last)

class GraspNetRealPointClouds(GraspNetPointClouds):
    def __init__(self, dataroot, mode, partition='train'):
        super(GraspNetRealPointClouds).__init__()
        self.partition = partition

        dataroot = os.path.join(dataroot, 'GraspNetPC/GraspNetPointClouds')

        # pdb.set_trace()

        DATA_DIR = os.path.join(dataroot, partition, "Real", mode) # mode can be 'kinect' or 'realsense'
        # read data
        xyzs_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.xyz')))

        self.pc_list = []
        self.lbl_list = []

        for xyz_path in xyzs_list:
            self.pc_list.append(xyz_path)
            self.lbl_list.append(int(xyz_path.split('/')[-2]))

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)


class GraspNetSynthetictPointClouds(GraspNetPointClouds):
    def __init__(self, dataroot, partition='train', device=None, use_density=True, use_drop=True, use_scan=True):
        super(GraspNetSynthetictPointClouds).__init__()
        self.partition = partition

        dataroot = os.path.join(dataroot, 'GraspNetPC/GraspNetPointClouds')

        # pdb.set_trace()

        if device == None:
            DATA_DIR_kinect = os.path.join(dataroot, partition, "Synthetic", "kinect")
            DATA_DIR_realsense = os.path.join(dataroot, partition, "Synthetic", "realsense")
            xyzs_list = sorted(glob.glob(os.path.join(DATA_DIR_kinect, '*', '*.xyz')))
            xyzs_list_realsense = sorted(glob.glob(os.path.join(DATA_DIR_realsense, '*', '*.xyz')))

            xyzs_list.extend(xyzs_list_realsense)
        elif device == 'kinect':
            DATA_DIR = os.path.join(dataroot, partition, "Synthetic", "kinect")
            xyzs_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.xyz')))
        elif device == 'realsense':
            DATA_DIR = os.path.join(dataroot, partition, "Synthetic", "realsense")
            xyzs_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.xyz')))

        self.pc_list = []
        self.lbl_list = []

        for xyz_path in xyzs_list:
            self.pc_list.append(xyz_path)
            self.lbl_list.append(int(xyz_path.split('/')[-2]))

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)



if __name__ == '__main__':
    root = '../../data/GraspNetPC-10/GraspNetPointClouds/'
    device = 'kinect'
    dataset = GraspNetSynthetictPointClouds(root, partition='train', device=device)