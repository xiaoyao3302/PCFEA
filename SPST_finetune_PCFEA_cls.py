import numpy as np
import random
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from utils.pc_utils import random_rotate_one_axis
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import argparse
import copy
import utils.log_SPST
from data.dataloader_GraspNetPC import GraspNetRealPointClouds, GraspNetSynthetictPointClouds
from data.dataloader_PointDA_initial import ScanNet, ModelNet, ShapeNet
from models.model import linear_DGCNN_model
import pdb

MAX_LOSS = 9 * (10 ** 9)


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
parser.add_argument('--dataroot', type=str, default='../data/', metavar='N', help='data path')
parser.add_argument('--out_path', type=str, default='./experiments/', help='log folder path')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers in dataloader')
parser.add_argument('--exp_name', type=str, default='test2', help='Name of the experiment')

# model
parser.add_argument('--model', type=str, default='DGCNN', choices=['PointNet++', 'DGCNN'], help='Model to use')
parser.add_argument('--num_class', type=int, default=10, help='number of classes per dataset')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--use_avg_pool', type=str2bool, default=False, help='Using average pooling & max pooling or max pooling only')

# training details
parser.add_argument('--epochs', type=int, default=10, help='number of episode per iteration to train')
parser.add_argument('--num_iterations', type=int, default=10, help='number of SPST iterations')
parser.add_argument('--src_dataset', type=str, default='Syn', choices=['Syn', 'Kin', 'RS', 'modelnet', 'shapenet', 'scannet'])
parser.add_argument('--trgt_dataset', type=str, default='Kin', choices=['Kin', 'RS', 'modelnet', 'shapenet', 'scannet'])
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='1',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--batch_size', type=int, default=12, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=12, metavar='batch_size', help='Size of test batch per domain')

# method
parser.add_argument('--base_threshold', type=float, default=0.8, help="base threshold to select target samples")
parser.add_argument('--use_SPL', type=str2bool, default=True, help='Using self paced self train or not')
parser.add_argument('--use_aug', type=str2bool, default=False, help='Using target augmentation or not (maybe can increase generalization)')
parser.add_argument('--save_iter_model_by_val', type=str2bool, default=True, help='Saving model by val or test in each iteration')
parser.add_argument('--mode_checkpoint', type=str, default='val', help='Using saved best model according to val or test')

# optimizer
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')

args = parser.parse_args()

# ==================
# init
# ==================
io = utils.log_SPST.IOStream(args)
io.cprint(str(args))

random.seed(1)
# np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')


# ==================
# Utils
# ==================
def split_set(dataset, domain, set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


class DataLoad(Dataset):
    def __init__(self, io, data):
        self.pc, self.aug_pc, self.label, self.real_label, self.num_data = data
        self.num_examples = len(self.pc)

        io.cprint("number of selected examples in train set: " + str(len(self.pc)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in train set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pc = np.copy(self.pc[item])
        aug_pc = np.copy(self.aug_pc[item])
        label = np.copy(self.label[item])
        real_label = np.copy(self.real_label[item])
        return (pc, aug_pc, label, real_label)

    def __len__(self):
        return len(self.pc)


def select_sample_by_conf(device, threshold, data_loader, model=None):
    pc_list = []
    aug_pc_list = []
    label_list = []
    real_label_list = []
    sfm = nn.Softmax(dim=1)

    total_number = 0

    with torch.no_grad():
        model.eval()
        for data_all in data_loader:
            data = data_all[1]
            labels = data_all[2]
            aug_data = data_all[3]
            data, labels, aug_data = data.to(device), labels.long().to(device), aug_data.to(device)

            if data.shape[2] < data.shape[1]:
                data = data.permute(0, 2, 1)        # data: B, C, N
            if aug_data.shape[2] < aug_data.shape[1]:
                aug_data = aug_data.permute(0, 2, 1)        # data: B, C, N

            batch_size = data.size()[0]
            total_number += batch_size

            logits = model(data)
            cls_pred = logits["pred"]
            cls_pred_sfm = sfm(cls_pred)
            cls_pred_conf, cls_pred_label = torch.max(cls_pred_sfm, 1)  # 2 * b

            index = 0
            for ii in range(batch_size):
                if cls_pred_conf[ii] > threshold:
                    # pdb.set_trace()
                    if len(pc_list) is 0:
                        pc_list = data[index].detach().cpu().unsqueeze(0)
                        label_list = cls_pred_label[index].detach().cpu().unsqueeze(0)
                        real_label_list = labels[index].detach().cpu().unsqueeze(0)
                        aug_pc_list = aug_data[index].detach().cpu().unsqueeze(0)
                    else:
                        pc_list = torch.cat((pc_list, data[index].detach().cpu().unsqueeze(0)), dim=0)
                        label_list = torch.cat((label_list, cls_pred_label[index].detach().cpu().unsqueeze(0)), dim=0)
                        real_label_list = torch.cat((real_label_list, labels[index].detach().cpu().unsqueeze(0)), dim=0)
                        aug_pc_list = torch.cat((aug_pc_list, aug_data[index].detach().cpu().unsqueeze(0)), dim=0)

                index += 1

    return pc_list, aug_pc_list, label_list, real_label_list, total_number


# ==================
# Data loader
# ==================

src_dataset = args.src_dataset
trgt_dataset = args.trgt_dataset

# source
if src_dataset == 'modelnet':
    src_trainset = ModelNet(io, args.dataroot, 'train')
    src_testset = ModelNet(io, args.dataroot, 'test')

elif src_dataset == 'shapenet':
    src_trainset = ShapeNet(io, args.dataroot, 'train')
    src_testset = ShapeNet(io, args.dataroot, 'test')

elif src_dataset == 'scannet':
    src_trainset = ScanNet(io, args.dataroot, 'train')
    src_testset = ScanNet(io, args.dataroot, 'test')

elif src_dataset == 'Syn':
    if trgt_dataset == 'RS':
        trgt_device = 'realsense'
    if trgt_dataset == 'Kin':
        trgt_device = 'kinect'
    src_trainset = GraspNetSynthetictPointClouds(args.dataroot, partition='train')
    src_testset = GraspNetRealPointClouds(args.dataroot, mode=trgt_device, partition='test')

elif src_dataset == 'Kin':
    src_trainset = GraspNetRealPointClouds(args.dataroot, mode='kinect', partition='train')
    src_testset = GraspNetRealPointClouds(args.dataroot, mode='kinect', partition='test')

elif src_dataset == 'RS':
    src_trainset = GraspNetRealPointClouds(args.dataroot, mode='realsense', partition='train')
    src_testset = GraspNetRealPointClouds(args.dataroot, mode='realsense', partition='test')

else:
    io.cprint('unknown src dataset')

# target
if trgt_dataset == 'modelnet':
    trgt_trainset = ModelNet(io, args.dataroot, 'train')
    trgt_testset = ModelNet(io, args.dataroot, 'test')

elif trgt_dataset == 'shapenet':
    trgt_trainset = ShapeNet(io, args.dataroot, 'train')
    trgt_testset = ShapeNet(io, args.dataroot, 'test')

elif trgt_dataset == 'scannet':
    trgt_trainset = ScanNet(io, args.dataroot, 'train')
    trgt_testset = ScanNet(io, args.dataroot, 'test')

elif trgt_dataset == 'Kin':
    trgt_trainset = GraspNetRealPointClouds(args.dataroot, mode='kinect', partition='train')
    trgt_testset = GraspNetRealPointClouds(args.dataroot, mode='kinect', partition='test')

elif trgt_dataset == 'RS':
    trgt_trainset = GraspNetRealPointClouds(args.dataroot, mode='realsense', partition='train')
    trgt_testset = GraspNetRealPointClouds(args.dataroot, mode='realsense', partition='test')

else:
    io.cprint('unknown trgt dataset')

src_train_sampler, src_valid_sampler = split_set(src_trainset, src_dataset, "source")
trgt_train_sampler, trgt_valid_sampler = split_set(trgt_trainset, trgt_dataset, "target")

# dataloaders for source and target
src_train_loader = DataLoader(src_trainset, num_workers=args.num_workers, batch_size=args.batch_size, sampler=src_train_sampler, drop_last=True)
src_val_loader = DataLoader(src_trainset, num_workers=args.num_workers, batch_size=args.test_batch_size, sampler=src_valid_sampler)
src_test_loader = DataLoader(src_testset, num_workers=args.num_workers, batch_size=args.test_batch_size)

trgt_train_loader = DataLoader(trgt_trainset, num_workers=args.num_workers, batch_size=args.batch_size, sampler=trgt_train_sampler, drop_last=True)
trgt_val_loader = DataLoader(trgt_trainset, num_workers=args.num_workers, batch_size=args.test_batch_size, sampler=trgt_valid_sampler)
trgt_test_loader = DataLoader(trgt_testset, num_workers=args.num_workers, batch_size=args.test_batch_size)


# ==================
# Init Model
# ==================
model = linear_DGCNN_model(args)
model = model.to(device)
io.cprint("------------------------------------------------------------------")
try:
    if args.mode_checkpoint == 'val':
        checkpoint = torch.load(args.out_path + '/' + args.src_dataset + '_' + args.trgt_dataset + '/'  + args.model + '/' + args.exp_name + '/save_best_by_val/model.pt')
    else:
        checkpoint = torch.load(args.out_path + '/' + args.src_dataset + '_' + args.trgt_dataset + '/'  + args.model + '/' + args.exp_name + '/save_best_by_test/model.pt')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    io.cprint('load saved model')
except:
    start_epoch = 0
    io.cprint('no saved model')

# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
best_model = copy.deepcopy(model)


# ==================
# Optimizer
# ==================
if args.optimizer == "SGD":
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
else:
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

scheduler = CosineAnnealingLR(opt, args.epochs)

criterion_cls = nn.CrossEntropyLoss()


# ==================
# Validation/test
# ==================
def test(loader, model=None, set_type="Target", partition="Val", epoch=0):
    # Run on cpu or gpu
    count = 0.0
    print_losses = {'cls': 0.0}
    batch_idx = 0

    with torch.no_grad():
        model.eval()

        test_pred = []
        test_true = []

        num_sample = 0

        for data_all in loader:
            data, labels = data_all[1], data_all[2]
            data, labels = data.to(device), labels.to(device).squeeze()

            if data.shape[0] == 1:
                labels = labels.unsqueeze(0)

            if data.shape[1] > data.shape[2]:
                data = data.permute(0, 2, 1)

            batch_size = data.size()[0]
            num_point = data.shape[-1]

            num_sample = num_sample + batch_size

            logits = model(data)
            loss = criterion_cls(logits["pred"], labels)
            print_losses['cls'] += loss.item() * batch_size

            # evaluation metrics
            preds = logits["pred"].max(dim=1)[1]

            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

            count += batch_size
            batch_idx += 1

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}

    test_acc = io.print_progress(set_type, partition, epoch, print_losses, test_true, test_pred)

    conf_mat = metrics.confusion_matrix(test_true, test_pred, labels=list(range(args.num_class))).astype(int)

    return test_acc, print_losses['cls'], conf_mat


# ==================
# Train
# ==================
# first test the performance of the loaded model
io.cprint("------------------------------------------------------------------") 
trgt_test_acc, trgt_test_loss, trgt_test_conf_mat = test(trgt_test_loader, model, "Target", "Test", 0)
io.cprint("------------------------------------------------------------------")
io.cprint("the performance of the loaded model is: %.4f" % (trgt_test_acc))

trgt_best_acc_by_val = trgt_test_acc
trgt_best_acc_by_test = trgt_test_acc

best_epoch_by_val = 0
best_epoch_by_test = 0

threshold_epoch = args.base_threshold

sfm = nn.Softmax(dim=1)

io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

for ii in range(args.num_iterations):

    # determine threshold
    if trgt_best_acc_by_test > 0.9:
        threshold_epoch = 0.95

    trgt_best_acc_by_val_by_iter = 0
    trgt_best_acc_by_test_by_iter = 0

    best_epoch_by_val_by_iter = 0
    best_epoch_by_test_by_iter = 0

    io.cprint("==================================================================") 
    io.cprint("iteration: %d, current threshold: %.4f" % (ii, threshold_epoch))
    io.cprint("------------------------------------------------------------------") 

    trgt_select_data = select_sample_by_conf(device, threshold_epoch, trgt_train_loader, model)
    trgt_new_data = DataLoad(io, trgt_select_data)
    trgt_new_train_loader = DataLoader(trgt_new_data, num_workers=args.num_workers, batch_size=args.batch_size, drop_last=True)
    io.cprint("------------------------------------------------------------------") 

    count = 0.0
    print_losses = {'cls': 0.0, 'total': 0.0}

    for epoch in range(args.epochs):

        model.train()

        for trgt_data_all in trgt_new_train_loader:

            opt.zero_grad()

            trgt_data, aug_trgt_data, trgt_label, trgt_real_label = trgt_data_all[0].to(device), trgt_data_all[1].to(device), trgt_data_all[2].long().to(device), trgt_data_all[3].long().to(device)

            if trgt_data.shape[1] > trgt_data.shape[2]:
                trgt_data = trgt_data.permute(0, 2, 1)
            if aug_trgt_data.shape[1] > aug_trgt_data.shape[2]:
                aug_trgt_data = aug_trgt_data.permute(0, 2, 1)

            batch_size = trgt_data.shape[0]
            num_point = trgt_data.shape[-1]

            # start training process
            if args.use_aug:
                trgt_logits = model(aug_trgt_data)
            else:
                trgt_logits = model(trgt_data)

            # ============== #
            # calculate loss #
            # ============== #
            trgt_feature = trgt_logits['feature']
            trgt_pred = trgt_logits['pred']
            trgt_pred_sfm = sfm(trgt_pred)

            cls_loss = criterion_cls(trgt_pred, trgt_label)
            total_loss = cls_loss

            print_losses['cls'] += total_loss.item() * batch_size
            print_losses['total'] += total_loss.item() * batch_size

            total_loss.backward()
            
            count += batch_size

            opt.step()

        scheduler.step()

        print_losses = {k: v * 1.0 / (count + 1e-6) for (k, v) in print_losses.items()}
        io.print_progress("Target", "Trn", epoch, print_losses)
        io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++") 

        # ===================
        # Test
        # ===================
        io.cprint("------------------------------------------------------------------")
        trgt_val_acc, trgt_val_loss, trgt_val_conf_mat = test(trgt_val_loader, model, "Target", "Val", epoch)
        io.cprint("------------------------------------------------------------------")
        trgt_test_acc, trgt_test_loss, trgt_test_conf_mat = test(trgt_test_loader, model, "Target", "Test", epoch)
        io.cprint("------------------------------------------------------------------")

        if trgt_val_acc > trgt_best_acc_by_val_by_iter:
            trgt_best_acc_by_val_by_iter = trgt_test_acc
            best_epoch_by_val_by_iter = epoch
            best_epoch_conf_mat_by_val_by_iter = trgt_test_conf_mat
            best_model_by_val_by_iter = copy.deepcopy(model)

        if trgt_test_acc > trgt_best_acc_by_test_by_iter:
            trgt_best_acc_by_test_by_iter = trgt_test_acc
            best_epoch_by_test_by_iter = epoch
            best_epoch_conf_mat_by_test_by_iter = trgt_test_conf_mat
            best_model_by_test_by_iter = copy.deepcopy(model)
    
        io.cprint("------------------------------------------------------------------")
        io.cprint("iteration: %d, epoch: %d, " % (ii, epoch))
        io.cprint("previous best target test accuracy saved by val during each iteration: %.4f" % (trgt_best_acc_by_val_by_iter))
        io.cprint("previous best target test accuracy saved by test during each iteration: %.4f" % (trgt_best_acc_by_test_by_iter))
        io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # update
    if args.save_iter_model_by_val:
        model = copy.deepcopy(trgt_best_acc_by_val_by_iter)
    else:
        model = copy.deepcopy(best_model_by_test_by_iter)

    if args.use_SPL:
        threshold_epoch += 0.01
    if threshold_epoch > 0.95:
        threshold_epoch = 0.95

    if trgt_best_acc_by_val_by_iter > trgt_best_acc_by_val:
        trgt_best_acc_by_val = trgt_best_acc_by_val_by_iter
        best_epoch_by_val = best_epoch_by_val_by_iter
        best_epoch_conf_mat_by_val = best_epoch_conf_mat_by_val_by_iter
        best_model_by_val = io.save_model(model, epoch, 'save_best_by_SPST_val')

    if trgt_best_acc_by_test_by_iter > trgt_best_acc_by_test:
        trgt_best_acc_by_test = trgt_best_acc_by_test_by_iter
        best_epoch_by_test = best_epoch_by_test_by_iter
        best_epoch_conf_mat_by_test = best_epoch_conf_mat_by_test_by_iter
        best_model_by_test = io.save_model(model, epoch, 'save_best_by_SPST_test')

io.cprint("Best model searched by val was found at epoch %d, target test accuracy: %.4f"
          % (best_epoch_by_val, trgt_best_acc_by_val))
io.cprint("Best test model confusion matrix:")
io.cprint('\n' + str(best_epoch_conf_mat_by_val))

io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

io.cprint("Best model searched by test was found at epoch %d, target test accuracy: %.4f"
          % (best_epoch_by_test, trgt_best_acc_by_test))
io.cprint("Best test model confusion matrix:")
io.cprint('\n' + str(best_epoch_conf_mat_by_test))

io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
