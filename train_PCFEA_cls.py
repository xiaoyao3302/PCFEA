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
import utils.log
from data.dataloader_GraspNetPC import GraspNetRealPointClouds, GraspNetSynthetictPointClouds
from data.dataloader_PointDA_initial import ScanNet, ModelNet, ShapeNet
from models.model import linear_DGCNN_model
from critic import PCFEALoss_no_mean, PCFEALoss, CalculateSelectedMean, CalculateSelectedCV, IDFALoss
from tensorboardX import SummaryWriter  
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
parser.add_argument('--epochs', type=int, default=100, help='number of episode to train')
parser.add_argument('--src_dataset', type=str, default='Syn', choices=['Syn', 'Kin', 'RS', 'modelnet', 'shapenet', 'scannet'])
parser.add_argument('--trgt_dataset', type=str, default='Kin', choices=['Kin', 'RS', 'modelnet', 'shapenet', 'scannet'])
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='1',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--batch_size', type=int, default=12, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=12, metavar='batch_size', help='Size of test batch per domain')

# method
parser.add_argument('--use_aug', type=str2bool, default=False, help='Using source augmentation or not')
parser.add_argument('--lambda_0', type=float, default=0.5, help='lambda in TSA')
parser.add_argument('--epoch_warmup', type=int, default=0, help='0: no warm up; only train a w/o DA method')
parser.add_argument('--selection_strategy', type=str, default='threshold', choices=['threshold', 'ratio'])
parser.add_argument('--use_gradual_src_threshold', type=str2bool, default=True, help='Using changing threshold to select source samples or not')
parser.add_argument('--use_gradual_trgt_threshold', type=str2bool, default=True, help='Using changing threshold to select target samples or not')
parser.add_argument('--mode_src_threshold', type=str, default='linear', choices=['linear', 'nonlinear'])
parser.add_argument('--mode_trgt_threshold', type=str, default='linear', choices=['linear', 'nonlinear'])
parser.add_argument('--src_threshold', type=float, default=0.0, help="threshold to select source samples, increasing")
parser.add_argument('--trgt_threshold', type=float, default=1.0, help="threshold to select target samples, decreasing")
parser.add_argument('--exp_k', type=float, default=0.1, help='parameter in exp in threshold')
parser.add_argument('--use_gradual_src_ratio', type=str2bool, default=True, help='Using changing ratio to select source samples or not')
parser.add_argument('--use_gradual_trgt_ratio', type=str2bool, default=True, help='Using changing ratio to select target samples or not')
parser.add_argument('--src_ratio', type=float, default=1.0, help="threshold to select source samples, increasing")
parser.add_argument('--trgt_ratio', type=float, default=1.0, help="threshold to select target samples, decreasing")
parser.add_argument('--period_update_pool', type=int, default=1, help='period to update the pool')
parser.add_argument('--use_model_eval', type=str2bool, default=True, help='Using the eval mode of the model or the train mode')
parser.add_argument('--loss_function', type=str, default='use_mean', choices=['no_mean', 'use_mean', 'CE'], help='use mean or not in our PCFEA loss or CE')
parser.add_argument('--use_EMA', type=str2bool, default=True, help='Using teacher model or not')
parser.add_argument('--EMA_update_warmup', type=str2bool, default=True, help='Update the teacher model in the warm up stage or not')
parser.add_argument('--EMA_decay', type=float, default=0.99, help='initial weight decay in EMA')
parser.add_argument('--use_src_IDFA', type=str2bool, default=False, help='Using the prototype alignment on the source data')
parser.add_argument('--use_trgt_IDFA', type=str2bool, default=False, help='Using the prototype alignment on the target data')
parser.add_argument('--tao', type=float, default=0.1, help='tao in prototype alignment')
parser.add_argument('--w_PCFEA', type=float, default=1.0, help='weight of PCFEA loss')
parser.add_argument('--w_src_IDFA', type=float, default=1.0, help='weight of source prototype alignment loss')
parser.add_argument('--w_trgt_IDFA', type=float, default=1.0, help='weight of target prototype alignment loss')
parser.add_argument('--save_distribution', type=str2bool, default=False, help='Save intermnediate domain distribution or not')

# optimizer
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')

args = parser.parse_args()

# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))

args.save_path = io.path
tb_dir = args.save_path
tb = SummaryWriter(log_dir=tb_dir)

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
# Read Data
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
              " valIDFAtion part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


# ==================
# Utils
# ==================

class DataLoad(Dataset):
    def __init__(self, io, data, len_dataset):
        self.idx, self.pc, self.label, self.aug_pc = data
        self.num_examples = len(self.pc)

        io.cprint("number of selected examples in the dataset is %d out of total %d samples" % (self.num_examples, len_dataset))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes of selected examples in the dataset: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        idx = np.copy(self.idx[item])
        pc = np.copy(self.pc[item])
        label = np.copy(self.label[item])
        aug_pc = np.copy(self.aug_pc[item])
        return (idx, pc, label, aug_pc)

    def __len__(self):
        return len(self.pc)
    

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
teacher_model = linear_DGCNN_model(args)

model = model.to(device)
teacher_model = teacher_model.to(device)

# detach the gradient of the teacher model
for p in teacher_model.parameters():
    p.requires_grad = False

# initialize teacher model
with torch.no_grad():
    for t_params, s_params in zip(teacher_model.parameters(), model.parameters()):
        t_params.data = s_params.data

# load model
try:
    checkpoint = torch.load(args.out_path + '/' + args.src_dataset + '_' + args.trgt_dataset + '/'  + args.model + '/' + args.exp_name + '/save_best_by_test/model.pt')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    io.cprint('load saved model')
except:
    start_epoch = 0
    io.cprint('no saved model')

# load teacher model
try:
    checkpoint = torch.load(args.out_path + '/' + args.src_dataset + '_' + args.trgt_dataset + '/'  + args.model + '/' + args.exp_name + '/teacher/model.pt')
    teacher_model.load_state_dict(checkpoint['model'])
    io.cprint('load saved teacher model')
except:
    start_epoch = 0
    io.cprint('no saved teacher model')

# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
    teacher_model = nn.DataParallel(teacher_model, args.gpus)

best_model = copy.deepcopy(model)
best_teacher_model = copy.deepcopy(teacher_model)


# ==================
# Optimizer
# ==================
if args.optimizer == "SGD":
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
else:
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

scheduler = CosineAnnealingLR(opt, args.epochs)

criterion_cls = nn.CrossEntropyLoss()

if args.loss_function == 'no_mean':
    criterion_PCFEA = PCFEALoss_no_mean(args.num_class)
elif args.loss_function == 'use_mean':
    criterion_PCFEA = PCFEALoss(args.num_class)
else:
    criterion_PCFEA = nn.CrossEntropyLoss()


# ==================
# ValIDFAtion/test
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
src_best_val_acc = 0
src_best_test_acc = 0

trgt_best_acc_by_src_val = 0
trgt_best_acc_by_src_test = 0
trgt_best_acc_by_trgt_val = 0
trgt_best_acc_by_trgt_test = 0

src_threshold = args.src_threshold
trgt_threshold = args.trgt_threshold
src_ratio = args.src_ratio
trgt_ratio = args.trgt_ratio

s_len = int(len(src_trainset))
t_len = int(len(trgt_trainset))

sfm = nn.Softmax(dim=1)

io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

count_src_selected_sample = int(s_len * 0.8)
count_trgt_selected_sample = 0

iters = 0

for epoch in range(args.epochs):
    io.cprint("epoch %d, " % (epoch))
    # -------------------------------------------------------------------------- #
    # selection stage
    # -------------------------------------------------------------------------- #

    # initialize counting indicators
    count_sample = 0.0

    if args.selection_strategy == 'threshold':
        if (epoch >= args.epoch_warmup) & ((epoch - args.epoch_warmup) % args.period_update_pool == 0):
            # initialize
            if args.use_gradual_src_threshold:
                # linear change threshold
                if args.mode_src_threshold == 'linear':
                    src_threshold += (1.0 / args.epochs * args.period_update_pool)
                # exp change threshold
                else:
                    src_threshold = (1 - math.exp(args.exp_k * (-(epoch - args.epoch_warmup)))) * args.src_threshold

            if args.use_gradual_trgt_threshold:
                # linear change threshold
                if args.mode_trgt_threshold == 'linear':
                    trgt_threshold -= (1.0 / args.epochs * args.period_update_pool)
                # exp change threshold
                else:
                    trgt_threshold = (1 - math.exp(args.exp_k * (-((args.epochs - args.epoch_warmup) - (epoch - args.epoch_warmup))))) * args.trgt_threshold

            if src_threshold >= 1.0:
                src_threshold = 1.0
            if trgt_threshold <= 0.0:
                trgt_threshold = 0.0

            count_src_selected_sample = 0.0
            count_trgt_selected_sample = 0.0

            # initialize memory module (containing both part of the source samples and target samples)
            memory_features = torch.zeros((s_len + t_len), 1024).to(device)
            memory_labels = torch.zeros(s_len + t_len).long().to(device)
            memory_indicator = torch.zeros(s_len + t_len).long().to(device)         # update this sample or not

            # calculate src mean
            src_features = torch.zeros(s_len, 1024).to(device)
            src_labels = torch.zeros(s_len).long().to(device)
            src_indicator = torch.zeros(s_len).long().to(device)

            # update memory
            if args.use_model_eval:
                model.eval()
            else:
                model.train()
            
            teacher_model.eval()
            
            # use source to update
            for src_data_all in src_train_loader:
                src_idx, src_data, src_label, aug_src_data = src_data_all[0], src_data_all[1].to(device), src_data_all[2].to(device), src_data_all[3].to(device)

                if src_data.shape[1] > src_data.shape[2]:
                    src_data = src_data.permute(0, 2, 1)
                
                if args.use_EMA:
                    src_logits = teacher_model(src_data)
                else:
                    src_logits = model(src_data)

                src_feature = src_logits['feature']
                src_pred = src_logits['pred']
                src_pred_sfm = sfm(src_pred)

                for ii in range(args.batch_size):
                    # select source sample
                    if (src_pred_sfm.max(dim=-1)[0][ii] > src_threshold) and (src_pred.max(dim=-1)[1][ii] == src_label[ii]):
                        memory_features[src_idx[ii]] = src_feature[ii].detach()
                        memory_labels[src_idx[ii]] = src_pred.max(dim=-1)[1][ii].detach()
                        memory_indicator[src_idx[ii]] = 1
                        count_src_selected_sample += 1
                
                for ii in range(args.batch_size):
                    # select source sample to update source memory
                    src_features[src_idx[ii]] = src_feature[ii].detach()
                    src_labels[src_idx[ii]] = src_label[ii].detach()
                    src_indicator[src_idx[ii]] = 1

            # use target to update
            for trgt_data_all in trgt_train_loader:
                trgt_idx, trgt_data, trgt_label, aug_trgt_data = trgt_data_all[0], trgt_data_all[1].to(device), trgt_data_all[2].to(device), trgt_data_all[3].to(device)
            
                if trgt_data.shape[1] > trgt_data.shape[2]:
                    trgt_data = trgt_data.permute(0, 2, 1)

                if args.use_EMA:
                    trgt_logits = teacher_model(trgt_data)
                else:
                    trgt_logits = model(trgt_data)

                trgt_feature = trgt_logits['feature']
                trgt_pred = trgt_logits['pred']
                trgt_pred_sfm = sfm(trgt_pred)

                for ii in range(args.batch_size):
                    # select target sample
                    if trgt_pred_sfm.max(dim=-1)[0][ii] > trgt_threshold:
                        memory_features[trgt_idx[ii] + s_len] = trgt_feature[ii].detach()
                        memory_labels[trgt_idx[ii] + s_len] = trgt_pred.max(dim=-1)[1][ii].detach()
                        memory_indicator[trgt_idx[ii] + s_len] = 1
                        count_trgt_selected_sample += 1

            mean_src = CalculateSelectedMean(src_features, src_labels, src_indicator, args.num_class)
            mean_pool = CalculateSelectedMean(memory_features, memory_labels, memory_indicator, args.num_class)
            cv_pool = CalculateSelectedCV(memory_features, memory_labels, memory_indicator, mean_pool, args.num_class)

            save_mean_src = mean_src.detach().cpu().numpy()
            save_mean_pool = mean_pool.detach().cpu().numpy()
            save_cv_pool = cv_pool.detach().cpu().numpy()

            if args.save_distribution:
                save_pool_path = args.out_path + '/' + args.src_dataset + '_' + args.trgt_dataset + '/'  + args.model + '/' + args.exp_name

                save_mean_src_name = save_pool_path + '/save_mean_src_' + str(epoch) + '.npy'
                save_mean_pool_name = save_pool_path + '/save_mean_pool_' + str(epoch) + '.npy'
                save_cv_pool_name = save_pool_path + '/save_cv_pool_src_' + str(epoch) + '.npy'

                np.save(save_mean_src_name, save_mean_src)
                np.save(save_mean_pool_name, save_mean_pool)
                np.save(save_cv_pool_name, save_cv_pool)

                io.cprint("successfully save memory banks in epoch %d" % epoch)

        io.cprint("------------------------------------------------------------------")
        io.cprint("current threshold for selecting source sample is %.4f" % (src_threshold))
        io.cprint("current threshold for selecting target sample is %.4f" % (trgt_threshold))
        io.cprint("------------------------------------------------------------------")
        io.cprint("select %d source samples out of %d samples" % (count_src_selected_sample, int(s_len * 0.8)))
        io.cprint("select %d target samples out of %d samples" % (count_trgt_selected_sample, int(t_len * 0.8)))
        io.cprint("------------------------------------------------------------------")
    
    elif args.selection_strategy == 'ratio':
        if (epoch >= args.epoch_warmup) & ((epoch - args.epoch_warmup) % args.period_update_pool == 0):
            # initialize
            if args.use_gradual_src_ratio:
                # linear change ratio
                src_ratio = epoch / (args.epoch_warmup - args.epochs) + 100 / (args.epochs - args.epoch_warmup)

            if args.use_gradual_trgt_ratio:
                # linear change ratio
                trgt_ratio = epoch / (args.epochs - args.epoch_warmup) + args.epoch_warmup / (args.epoch_warmup - args.epochs)

            if src_ratio <= 0.0:
                src_ratio = 0.0
            if trgt_ratio >= 1.0:
                trgt_ratio = 1.0

            count_src_selected_sample = 0.0
            count_trgt_selected_sample = 0.0

            # initialize memory module (containing both part of the source samples and target samples)
            memory_features = torch.zeros((s_len + t_len), 1024).to(device)
            memory_labels = torch.zeros(s_len + t_len).long().to(device)
            memory_indicator = torch.zeros(s_len + t_len).long().to(device)         # update this sample or not

            # collect source data and then filter the data according to the confidence score
            save_src_idx = []
            save_src_data = []
            save_src_label = []
            save_src_feature = []
            save_aug_src_data = []
            save_src_confidence = []

            # collect target data and then filter the data according to the confidence score
            save_trgt_idx = []
            save_trgt_data = []
            save_trgt_label = []
            save_trgt_feature = []
            save_aug_trgt_data = []
            save_trgt_confidence = []

            for cc in range(args.num_class):
                # source
                save_src_idx.append([])
                save_src_data.append([])
                save_src_label.append([])
                save_src_feature.append([])
                save_aug_src_data.append([])
                save_src_confidence.append([])

                # target
                save_trgt_idx.append([])
                save_trgt_data.append([])
                save_trgt_label.append([])
                save_trgt_feature.append([])
                save_aug_trgt_data.append([])
                save_trgt_confidence.append([])

            # calculate src mean
            src_features = torch.zeros(s_len, 1024).to(device)
            src_labels = torch.zeros(s_len).long().to(device)
            src_indicator = torch.zeros(s_len).long().to(device)

            # update memory
            if args.use_model_eval:
                model.eval()
            else:
                model.train()
            
            teacher_model.eval()
            
            # use source to update
            for src_data_all in src_train_loader:
                src_idx, src_data, src_label, aug_src_data = src_data_all[0], src_data_all[1].to(device), src_data_all[2].to(device), src_data_all[3].to(device)

                if src_data.shape[1] > src_data.shape[2]:
                    src_data = src_data.permute(0, 2, 1)
                
                if args.use_EMA:
                    src_logits = teacher_model(src_data)
                else:
                    src_logits = model(src_data)

                src_feature = src_logits['feature']
                src_pred = src_logits['pred']
                src_pred_sfm = sfm(src_pred)
                src_pred_confidence = src_pred_sfm.max(dim=-1)[0]
                src_pred_pseudo_label = src_pred_sfm.max(dim=-1)[1]

                for ii in range(args.batch_size):
                    # select source sample
                    if len(save_src_data[src_label[ii]]) == 0:
                        save_src_idx[src_label[ii]] = src_idx[ii].detach().cpu().unsqueeze(0)
                        save_src_data[src_label[ii]] = src_data[ii].detach().cpu().unsqueeze(0)
                        save_src_label[src_label[ii]] = src_label[ii].detach().cpu().unsqueeze(0)
                        save_src_feature[src_label[ii]] = src_feature[ii].detach().cpu().unsqueeze(0)
                        save_aug_src_data[src_label[ii]] = aug_src_data[ii].detach().cpu().unsqueeze(0)
                        save_src_confidence[src_label[ii]] = src_pred_confidence[ii].detach().cpu().unsqueeze(0)

                    else:
                        save_src_idx[src_label[ii]] = torch.cat((save_src_idx[src_label[ii]], src_idx[ii].detach().cpu().unsqueeze(0)), dim=0)
                        save_src_data[src_label[ii]] = torch.cat((save_src_data[src_label[ii]], src_data[ii].detach().cpu().unsqueeze(0)), dim=0)
                        save_src_label[src_label[ii]] = torch.cat((save_src_label[src_label[ii]], src_label[ii].detach().cpu().unsqueeze(0)), dim=0)
                        save_src_feature[src_label[ii]] = torch.cat((save_src_feature[src_label[ii]], src_feature[ii].detach().cpu().unsqueeze(0)), dim=0)
                        save_aug_src_data[src_label[ii]] = torch.cat((save_aug_src_data[src_label[ii]], aug_src_data[ii].detach().cpu().unsqueeze(0)), dim=0)
                        save_src_confidence[src_label[ii]] = torch.cat([save_src_confidence[src_label[ii]], src_pred_confidence[ii].detach().cpu().unsqueeze(0)], dim=0)

                for ii in range(args.batch_size):
                    # select source sample to update source memory
                    src_features[src_idx[ii]] = src_feature[ii].detach()
                    src_labels[src_idx[ii]] = src_label[ii].detach()
                    src_indicator[src_idx[ii]] = 1

            # use target to update
            for trgt_data_all in trgt_train_loader:
                trgt_idx, trgt_data, trgt_label, aug_trgt_data = trgt_data_all[0], trgt_data_all[1].to(device), trgt_data_all[2].to(device), trgt_data_all[3].to(device)
            
                if trgt_data.shape[1] > trgt_data.shape[2]:
                    trgt_data = trgt_data.permute(0, 2, 1)

                if args.use_EMA:
                    trgt_logits = teacher_model(trgt_data)
                else:
                    trgt_logits = model(trgt_data)

                trgt_feature = trgt_logits['feature']
                trgt_pred = trgt_logits['pred']
                trgt_pred_sfm = sfm(trgt_pred)
                trgt_pred_confidence = trgt_pred_sfm.max(dim=-1)[0]
                trgt_pred_pseudo_label = trgt_pred_sfm.max(dim=-1)[1]

                for ii in range(args.batch_size):
                    # select target sample
                    if len(save_trgt_data[trgt_pred_pseudo_label[ii]]) == 0:
                        save_trgt_idx[trgt_pred_pseudo_label[ii]] = trgt_idx[ii].detach().cpu().unsqueeze(0)
                        save_trgt_data[trgt_pred_pseudo_label[ii]] = trgt_data[ii].detach().cpu().unsqueeze(0)
                        save_trgt_label[trgt_pred_pseudo_label[ii]] = trgt_label[ii].detach().cpu().unsqueeze(0)
                        save_trgt_feature[trgt_pred_pseudo_label[ii]] = trgt_feature[ii].detach().cpu().unsqueeze(0)
                        save_aug_trgt_data[trgt_pred_pseudo_label[ii]] = aug_trgt_data[ii].detach().cpu().unsqueeze(0)
                        save_trgt_confidence[trgt_pred_pseudo_label[ii]] = trgt_pred_confidence[ii].detach().cpu().unsqueeze(0)

                    else:
                        save_trgt_idx[trgt_pred_pseudo_label[ii]] = torch.cat((save_trgt_idx[trgt_pred_pseudo_label[ii]], trgt_idx[ii].detach().cpu().unsqueeze(0)), dim=0)
                        save_trgt_data[trgt_pred_pseudo_label[ii]] = torch.cat((save_trgt_data[trgt_pred_pseudo_label[ii]], trgt_data[ii].detach().cpu().unsqueeze(0)), dim=0)
                        save_trgt_label[trgt_pred_pseudo_label[ii]] = torch.cat((save_trgt_label[trgt_pred_pseudo_label[ii]], trgt_label[ii].detach().cpu().unsqueeze(0)), dim=0)
                        save_trgt_feature[trgt_pred_pseudo_label[ii]] = torch.cat((save_trgt_feature[trgt_pred_pseudo_label[ii]], trgt_feature[ii].detach().cpu().unsqueeze(0)), dim=0)
                        save_aug_trgt_data[trgt_pred_pseudo_label[ii]] = torch.cat((save_aug_trgt_data[trgt_pred_pseudo_label[ii]], aug_trgt_data[ii].detach().cpu().unsqueeze(0)), dim=0)
                        save_trgt_confidence[trgt_pred_pseudo_label[ii]] = torch.cat([save_trgt_confidence[trgt_pred_pseudo_label[ii]], trgt_pred_confidence[ii].detach().cpu().unsqueeze(0)], dim=0)

            # initialize memory module (containing both part of the source samples and target samples)
            memory_features = torch.zeros((s_len + t_len), 1024).to(device)
            memory_labels = torch.zeros(s_len + t_len).long().to(device)
            memory_indicator = torch.zeros(s_len + t_len).long().to(device)         # update this sample or not

            # sort the confidence score
            for cc in range(args.num_class):
                # source
                num_src_sample_cat = len(save_src_data[cc])
                if num_src_sample_cat > 0:
                    # sort the confidence score and keep a proportion of the sample
                    src_sort_confidence_value_cat, src_sort_confidence_idx_cat = save_src_confidence[cc].sort(descending=True)
                    num_keep_src_sample_cat = int(num_src_sample_cat * src_ratio)
                    keep_src_idx_cat = src_sort_confidence_idx_cat[:num_keep_src_sample_cat]        # keep index (search in save_src_data[cc])
                
                    memory_features[save_src_idx[cc][keep_src_idx_cat]] = save_src_feature[cc][keep_src_idx_cat].to(device)
                    memory_labels[save_src_idx[cc][keep_src_idx_cat]] = save_src_label[cc][keep_src_idx_cat].to(device)
                    memory_indicator[save_src_idx[cc][keep_src_idx_cat]] = 1
                    count_src_selected_sample += num_keep_src_sample_cat

                # target
                num_trgt_sample_cat = len(save_trgt_data[cc])
                if num_trgt_sample_cat > 0:
                    # sort the confidence score and keep a proportion of the sample
                    trgt_sort_confidence_value_cat, trgt_sort_confidence_idx_cat = save_trgt_confidence[cc].sort(descending=True)
                    num_keep_trgt_sample_cat = int(num_trgt_sample_cat * trgt_ratio)
                    keep_trgt_idx_cat = trgt_sort_confidence_idx_cat[:num_keep_trgt_sample_cat]        # keep index (search in save_trgt_data[cc])
                
                    memory_features[save_trgt_idx[cc][keep_trgt_idx_cat] + s_len] = save_trgt_feature[cc][keep_trgt_idx_cat].to(device)
                    memory_labels[save_trgt_idx[cc][keep_trgt_idx_cat] + s_len] = save_trgt_label[cc][keep_trgt_idx_cat].to(device)
                    memory_indicator[save_trgt_idx[cc][keep_trgt_idx_cat] + s_len] = 1
                    count_trgt_selected_sample += num_keep_trgt_sample_cat

            mean_src = CalculateSelectedMean(src_features, src_labels, src_indicator, args.num_class)
            mean_pool = CalculateSelectedMean(memory_features, memory_labels, memory_indicator, args.num_class)
            cv_pool = CalculateSelectedCV(memory_features, memory_labels, memory_indicator, mean_pool, args.num_class)

            save_mean_src = mean_src.detach().cpu().numpy()
            save_mean_pool = mean_pool.detach().cpu().numpy()
            save_cv_pool = cv_pool.detach().cpu().numpy()

            if args.save_distribution:
                save_pool_path = args.out_path + '/' + args.src_dataset + '_' + args.trgt_dataset + '/'  + args.model + '/' + args.exp_name

                save_mean_src_name = save_pool_path + '/save_mean_src_' + str(epoch) + '.npy'
                save_mean_pool_name = save_pool_path + '/save_mean_pool_' + str(epoch) + '.npy'
                save_cv_pool_name = save_pool_path + '/save_cv_pool_src_' + str(epoch) + '.npy'

                np.save(save_mean_src_name, save_mean_src)
                np.save(save_mean_pool_name, save_mean_pool)
                np.save(save_cv_pool_name, save_cv_pool)

                io.cprint("successfully save memory banks in epoch %d" % epoch)

        io.cprint("------------------------------------------------------------------")
        io.cprint("current selecting ratio for selecting source sample is %.4f" % (src_ratio))
        io.cprint("current selecting ratio for selecting target sample is %.4f" % (trgt_ratio))
        io.cprint("------------------------------------------------------------------")
        io.cprint("select %d source samples out of %d samples" % (count_src_selected_sample, int(s_len * 0.8)))
        io.cprint("select %d target samples out of %d samples" % (count_trgt_selected_sample, int(t_len * 0.8)))
        io.cprint("------------------------------------------------------------------")


    # -------------------------------------------------------------------------- #
    # start training
    # -------------------------------------------------------------------------- #

    Lambda = args.lambda_0 * (epoch / args.epochs)

    batch_idx = 1

    # initialize loss structures for saving epoch stats
    print_losses = {'total': 0.0, 'cls': 0.0, 'PCFEA': 0.0}
    if args.use_src_IDFA:
        print_losses['src_IDFA'] = 0.0
    if args.use_trgt_IDFA:
        print_losses['trgt_IDFA'] = 0.0

    # initialize model status
    model.train()

    for src_data_all, trgt_data_all in zip(src_train_loader, trgt_train_loader):

        opt.zero_grad()

        src_idx, src_data, src_label, aug_src_data = src_data_all[0], src_data_all[1].to(device), src_data_all[2].to(device), src_data_all[3].to(device)
        trgt_idx, trgt_data, trgt_label, aug_trgt_data = trgt_data_all[0], trgt_data_all[1].to(device), trgt_data_all[2].to(device), trgt_data_all[3].to(device)
        
        if src_data.shape[1] > src_data.shape[2]:
            src_data = src_data.permute(0, 2, 1)
        if aug_src_data.shape[1] > aug_src_data.shape[2]:
            aug_src_data = aug_src_data.permute(0, 2, 1)

        if trgt_data.shape[1] > trgt_data.shape[2]:
            trgt_data = trgt_data.permute(0, 2, 1)
        if aug_trgt_data.shape[1] > aug_trgt_data.shape[2]:
            aug_trgt_data = aug_trgt_data.permute(0, 2, 1)

        batch_size = src_data.shape[0]
        num_point = src_data.shape[-1]

        # start training process
        if args.use_aug:
            src_logits = model(aug_src_data)
            trgt_logits = model(aug_trgt_data)
        else:
            src_logits = model(src_data)
            trgt_logits = model(trgt_data)

        # ============== #
        # calculate loss #
        # ============== #

        total_loss = 0.0
        
        src_feature = src_logits['feature']
        src_pred = src_logits['pred']
        src_pred_sfm = sfm(src_pred)

        trgt_feature = trgt_logits['feature']
        trgt_pred = trgt_logits['pred']
        trgt_pred_sfm = sfm(trgt_pred)

        if epoch < args.epoch_warmup:
            cls_loss = criterion_cls(src_pred, src_label)
            total_loss = total_loss + cls_loss

            print_losses['cls'] += cls_loss.item() * batch_size
            print_losses['total'] += cls_loss.item() * batch_size
        
        else:
            # estimate the mean and covariance
            class_num = args.num_class

            if args.loss_function == 'no_mean':
                PCFEA_loss = criterion_PCFEA(model.cls, src_feature, src_pred, src_label, Lambda, cv_pool)
            elif args.loss_function == 'use_mean':
                PCFEA_loss = criterion_PCFEA(model.cls, src_feature, src_pred, src_label, Lambda, mean_pool, mean_src, cv_pool)
            else:
                # CE loss
                PCFEA_loss = criterion_PCFEA(src_pred, src_label)
            total_loss = total_loss + args.w_PCFEA * PCFEA_loss

            print_losses['PCFEA'] += PCFEA_loss.item() * batch_size
            print_losses['total'] += PCFEA_loss.item() * batch_size

            # prototype alignment on source
            if args.use_src_IDFA:
                src_proto_loss = IDFALoss(mean_pool.detach(), src_feature, src_label, args.tao)
                total_loss = total_loss + args.w_src_IDFA * src_proto_loss

                print_losses['src_IDFA'] += src_proto_loss.item() * batch_size
                print_losses['total'] += src_proto_loss.item() * batch_size

            # prototype alignment on target
            if args.use_trgt_IDFA:
                trgt_proto_loss = IDFALoss(mean_pool.detach(), trgt_feature, trgt_pred.max(dim=-1)[1], args.tao)
                total_loss = total_loss + args.w_trgt_IDFA * trgt_proto_loss

                print_losses['trgt_IDFA'] += trgt_proto_loss.item() * batch_size
                print_losses['total'] += trgt_proto_loss.item() * batch_size
                                    
        total_loss.backward()
        opt.step()
        batch_idx += 1
        count_sample += batch_size

        # log
        iters += batch_size

        tb.add_scalar('train_loss_total', print_losses['total'] / (batch_idx+1), iters)
        tb.add_scalar('train_loss_cls', print_losses['cls'] / (batch_idx+1), iters)
        tb.add_scalar('train_loss_PCFEA', print_losses['PCFEA'] / (batch_idx+1), iters)

        # update the teacher model per batch
        if epoch < args.epoch_warmup:
            if args.EMA_update_warmup:
                EMA_decay = args.EMA_decay
            else:
                EMA_decay = 0.0             # student model only
        else:
            EMA_decay = args.EMA_decay

        for t_params, s_params in zip(teacher_model.parameters(), model.parameters()):
            t_params.data = t_params.data * EMA_decay + s_params.data * (1 - EMA_decay)

    scheduler.step()

    # print progress
    print_losses = {k: v * 1.0 / (count_sample + 1e-6) for (k, v) in print_losses.items()}
    io.print_progress("", "Trn", epoch, print_losses)

    # ===================
    # Test
    # ===================
    io.cprint("------------------------------------------------------------------")
    src_val_acc, src_val_loss, src_val_conf_mat = test(src_val_loader, model, "Source", "Val", epoch)
    io.cprint("------------------------------------------------------------------")
    src_test_acc, src_test_loss, src_test_conf_mat = test(src_test_loader, model, "Source", "Test", epoch)
    io.cprint("------------------------------------------------------------------")
    trgt_val_acc, trgt_val_loss, trgt_val_conf_mat = test(trgt_val_loader, model, "Target", "Val", epoch)
    io.cprint("------------------------------------------------------------------")
    trgt_test_acc, trgt_test_loss, trgt_test_conf_mat = test(trgt_test_loader, model, "Target", "Test", epoch)
    io.cprint("------------------------------------------------------------------")

    if src_val_acc > src_best_val_acc:
        src_best_val_acc = src_val_acc
        trgt_best_acc_by_src_val = trgt_test_acc
        best_epoch_by_src_val = epoch
        best_model_by_src_val = io.save_model(model, epoch, 'save_best_by_src_val')

    if src_test_acc > src_best_test_acc:
        src_best_test_acc = src_test_acc
        trgt_best_acc_by_src_test = trgt_test_acc
        best_epoch_by_src_test = epoch
        best_model_by_src_test = io.save_model(model, epoch, 'save_best_by_src_test')

    if trgt_val_acc > trgt_best_acc_by_trgt_val:
        trgt_best_acc_by_trgt_val = trgt_test_acc
        best_epoch_by_trgt_val = epoch
        best_model_by_trgt_val = io.save_model(model, epoch, 'save_best_by_trgt_val')

    if trgt_test_acc > trgt_best_acc_by_trgt_test:
        trgt_best_acc_by_trgt_test = trgt_test_acc
        best_epoch_by_trgt_test = epoch
        best_model_by_trgt_test = io.save_model(model, epoch, 'save_best_by_trgt_test')
    
    best_teacher_model = io.save_model(model, epoch, 'teacher')

    io.cprint("------------------------------------------------------------------")
    io.cprint("previous best source val accuracy: %.4f" % (src_best_val_acc))
    io.cprint("previous best source test accuracy: %.4f" % (src_best_test_acc))
    io.cprint("previous best target test accuracy saved by src val: %.4f" % (trgt_best_acc_by_src_val))
    io.cprint("previous best target test accuracy saved by src test: %.4f" % (trgt_best_acc_by_src_test))
    io.cprint("previous best target test accuracy saved by trgt val: %.4f" % (trgt_best_acc_by_trgt_val))
    io.cprint("previous best target test accuracy saved by trgt test: %.4f" % (trgt_best_acc_by_trgt_test))
    io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    tb.add_scalar('source_test_accuracy', src_test_acc, epoch)
    tb.add_scalar('target_test_accuracy', trgt_test_acc, epoch)


io.cprint("Best model searched by src val was found at epoch %d, target test accuracy: %.4f"
          % (best_epoch_by_src_val, trgt_best_acc_by_src_val))

io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

io.cprint("Best model searched by src test was found at epoch %d, target test accuracy: %.4f"
          % (best_epoch_by_src_test, trgt_best_acc_by_src_test))

io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

io.cprint("Best model searched by trgt val was found at epoch %d, target test accuracy: %.4f"
          % (best_epoch_by_trgt_val, trgt_best_acc_by_trgt_val))

io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

io.cprint("Best model searched by trgt test was found at epoch %d, target test accuracy: %.4f"
          % (best_epoch_by_trgt_test, trgt_best_acc_by_trgt_test))

io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
io.cprint("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
