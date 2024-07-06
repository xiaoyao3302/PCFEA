import torch
import torch.utils.data as data
import os
import sys
import h5py
import numpy as np
from multiprocessing.dummy import Pool
from torchvision import transforms
import glob
import random
import threading
import time
from utils.pc_utils import *
from augmentation import density, drop_hole, p_scan
import pdb


class PaddingData(data.Dataset):
    def __init__(self, pc_root, status='train', swapax=False, pc_input_num=2048):
        super(PaddingData, self).__init__()

        self.status = status

        self.pc_list = []
        self.lbl_list = []
        self.transforms = transforms.Compose(
            [
                PointcloudToTensor(),
                PointcloudScale(),
                PointcloudRotate(),
                PointcloudRotatePerturbation(),
                PointcloudTranslate(),
                PointcloudJitter(),
            ]
        )
        self.pc_input_num = pc_input_num

        categorys = glob.glob(os.path.join(pc_root, '*'))
        categorys = [c.split(os.path.sep)[-1] for c in categorys]
        categorys = sorted(categorys)
        print(categorys)

        if status == 'train':
            npy_list = glob.glob(os.path.join(pc_root, '*', 'train', '*.npy'))
        else:
            npy_list = glob.glob(os.path.join(pc_root, '*', 'test', '*.npy'))

        for idx, _dir in enumerate(npy_list):
            print("\r%d/%d" % (idx, len(npy_list)), end="")
            pc = np.load(_dir).astype(np.float32)
            if swapax:
                pc[:, 1] = pc[:, 2] + pc[:, 1]
                pc[:, 2] = pc[:, 1] - pc[:, 2]
                pc[:, 1] = pc[:, 1] - pc[:, 2]
            self.pc_list.append(pc)
            self.lbl_list.append(categorys.index(_dir.split('/')[-3]))
        print()

        print(f'{status} data num: {len(self.pc_list)}')

    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        pc = self.pc_list[idx]      # 2048, 3
        pc = normal_pc(pc)

        pn = min(pc.shape[0], self.pc_input_num)
        if self.status == 'train':
            pc_aug = pc
            if np.random.random() > 0.5:
                pc_aug = density(pc_aug, num_point=2048)
            if np.random.random() > 0.5:
                pc_aug = drop_hole(pc_aug, num_point=2048)
            if np.random.random() > 0.5:
                pc_aug = p_scan(pc_aug, num_point=2048)

            pc_aug = self.transforms(pc_aug)
            pc_aug = pc_aug.numpy()

            pc = self.transforms(pc)
            pc = pc.numpy()
        else:
            pc_aug = pc

        if pn < self.pc_input_num:
            pc = np.append(pc, np.zeros((self.pc_input_num - pc.shape[0], 3)), axis=0)
            pc_aug = np.append(pc_aug, np.zeros((self.pc_input_num - pc_aug.shape[0], 3)), axis=0)
        pc = pc[:self.pc_input_num]
        pc_aug = pc_aug[:self.pc_input_num]

        return (idx, pc, lbl, pc_aug)

    def __len__(self):
        return len(self.pc_list)


if __name__ == '__main__':
    root = '../../data/MetaSets/'
    dataset = 'scanobjectnn_9'
    data_root = root + dataset
    dataset = PaddingData(data_root, status='train')
    print(dataset[1])
