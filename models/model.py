# add a average feature cls
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.trans_norm import TransNorm2d
import pdb
import argparse
from models.pointnet_util import PointNetSetAbstraction

K = 20


def index_points(points, idx):
    '''

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    '''
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class Mapping(nn.Module):
    def __init__(self, input_channel, hidden_channel, output_channel):
        super(Mapping, self).__init__()
        self.fc1 = nn.Conv1d(input_channel, hidden_channel, 1)
        self.fc2 = nn.Conv1d(hidden_channel, output_channel, 1)
        self.bn1 = nn.BatchNorm1d(hidden_channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        x = x / torch.norm(x, p=2, dim=-2, keepdim=True)
        return x


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # pdb.set_trace()
    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, args, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # Run on cpu or gpu
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # matrix [k*num_points*batch_size,3]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


def l2_norm(input, axit=1):
    norm = torch.norm(input, 2, axit, True)
    output = torch.div(input, norm)
    return output


class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu', bias=True):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                # nn.BatchNorm2d(out_ch),
                # nn.InstanceNorm2d(out_ch),
                TransNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                # nn.BatchNorm2d(out_ch),
                # nn.InstanceNorm2d(out_ch),
                TransNorm2d(out_ch),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False, activation='relu', bias=True):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                nn.LayerNorm(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                self.ac
            )

    def forward(self, x):
        x = l2_norm(x, 1)
        x = self.fc(x)
        return x


class transform_net(nn.Module):
    ''' Input (XYZ) Transform Net, input is BxNx3 gray image
        Return: Transformation matrix of size 3xK '''

    def __init__(self, args, in_ch, out=3):
        super(transform_net, self).__init__()
        self.K = out
        self.args = args

        activation = 'leakyrelu' if args.model == 'DGCNN' else 'relu'
        bias = False if args.model == 'DGCNN' else True

        self.conv2d1 = conv_2d(in_ch, 64, kernel=1, activation=activation, bias=bias)
        self.conv2d2 = conv_2d(64, 128, kernel=1, activation=activation, bias=bias)
        self.conv2d3 = conv_2d(128, 1024, kernel=1, activation=activation, bias=bias)
        self.fc1 = fc_layer(1024, 512, activation=activation, bias=bias, bn=True)
        self.fc2 = fc_layer(512, 256, activation=activation, bn=True)
        self.fc3 = nn.Linear(256, out * out)

    def forward(self, x):
        device = x.device

        x = self.conv2d1(x)
        x = self.conv2d2(x)
        if self.args.model == 'DGCNN':
            x = x.max(dim=-1, keepdim=False)[0]
            x = torch.unsqueeze(x, dim=3)
        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.K).view(1, self.K * self.K).repeat(x.size(0), 1)
        iden = iden.to(device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x


class DGCNN_encoder(nn.Module):
    def __init__(self, args):
        super(DGCNN_encoder, self).__init__()
        self.args = args
        self.k = K
        self.use_avg_pool = args.use_avg_pool

        self.input_transform_net = transform_net(args, 6, 3)

        self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64 * 2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64 * 2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128 * 2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        if self.use_avg_pool:
            # use avepooling + maxpooling
            self.conv5 = nn.Conv1d(num_f_prev, 512, kernel_size=1, bias=False)
            self.bn5 = nn.BatchNorm1d(512)
        else:
            # use only maxpooling
            self.conv5 = nn.Conv1d(num_f_prev, 1024, kernel_size=1, bias=False)
            self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        cls_logits = {}

        x = get_graph_feature(x, self.args, k=self.k)  # x: [b, 6, 1024, 20]
        x = self.conv1(x)  # x: [b, 64, 1024, 20]
        x1 = x.max(dim=-1, keepdim=False)[0]            # B, 64, 1024

        x = get_graph_feature(x1, self.args, k=self.k)      # [b, 128, 1024, 20]
        x = self.conv2(x)                                   # [b, 64, 1024, 20]
        x2 = x.max(dim=-1, keepdim=False)[0]                # [b, 64, 1024]

        x = get_graph_feature(x2, self.args, k=self.k)      # [b, 128, 1024, 20]
        x = self.conv3(x)                                   # [b, 128, 1024, 20]
        x3 = x.max(dim=-1, keepdim=False)[0]                # [b, 128, 1024]

        x = get_graph_feature(x3, self.args, k=self.k)      # [b, 256, 1024, 20]
        x = self.conv4(x)                                   # [b, 256, 1024, 20]
        x4 = x.max(dim=-1, keepdim=False)[0]                # [b, 256, 1024]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)          # [b, 512, 1024]

        if self.use_avg_pool:
            x5 = self.conv5(x_cat)                          # [b, 512, 1024]
            x5 = F.leaky_relu(self.bn5(x5), negative_slope=0.2)
            x5_1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
            x5_2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)
            x5_pool = torch.cat((x5_1, x5_2), 1)
        else:
            x5 = self.conv5(x_cat)                          # [b, 512, 1024]
            x5 = F.leaky_relu(self.bn5(x5), negative_slope=0.2)
            x5_pool = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)

        x = x5_pool

        cls_logits['feature'] = x
        
        return cls_logits


class DGCNN_model(nn.Module):
    def __init__(self, args):
        super(DGCNN_model, self).__init__()
        self.args = args
        self.k = K
        self.use_avg_pool = args.use_avg_pool

        self.input_transform_net = transform_net(args, 6, 3)

        self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64 * 2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64 * 2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128 * 2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        if self.use_avg_pool:
            # use avepooling + maxpooling
            self.conv5 = nn.Conv1d(num_f_prev, 512, kernel_size=1, bias=False)
            self.bn5 = nn.BatchNorm1d(512)
        else:
            # use only maxpooling
            self.conv5 = nn.Conv1d(num_f_prev, 1024, kernel_size=1, bias=False)
            self.bn5 = nn.BatchNorm1d(1024)
        
        self.cls = class_classifier(args, 1024, args.num_class)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        cls_logits = {}

        x = get_graph_feature(x, self.args, k=self.k)  # x: [b, 6, 1024, 20]
        x = self.conv1(x)  # x: [b, 64, 1024, 20]
        x1 = x.max(dim=-1, keepdim=False)[0]            # B, 64, 1024

        x = get_graph_feature(x1, self.args, k=self.k)      # [b, 128, 1024, 20]
        x = self.conv2(x)                                   # [b, 64, 1024, 20]
        x2 = x.max(dim=-1, keepdim=False)[0]                # [b, 64, 1024]

        x = get_graph_feature(x2, self.args, k=self.k)      # [b, 128, 1024, 20]
        x = self.conv3(x)                                   # [b, 128, 1024, 20]
        x3 = x.max(dim=-1, keepdim=False)[0]                # [b, 128, 1024]

        x = get_graph_feature(x3, self.args, k=self.k)      # [b, 256, 1024, 20]
        x = self.conv4(x)                                   # [b, 256, 1024, 20]
        x4 = x.max(dim=-1, keepdim=False)[0]                # [b, 256, 1024]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)          # [b, 512, 1024]

        if self.use_avg_pool:
            x5 = self.conv5(x_cat)                          # [b, 512, 1024]
            x5 = F.leaky_relu(self.bn5(x5), negative_slope=0.2)
            x5_1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
            x5_2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)
            x5_pool = torch.cat((x5_1, x5_2), 1)
        else:
            x5 = self.conv5(x_cat)                          # [b, 512, 1024]
            x5 = F.leaky_relu(self.bn5(x5), negative_slope=0.2)
            x5_pool = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)

        x = x5_pool

        cls_logits['feature'] = x

        cls_logits['pred'] = self.cls(x)
        
        return cls_logits


class linear_DGCNN_model(nn.Module):
    def __init__(self, args):
        super(linear_DGCNN_model, self).__init__()
        self.args = args
        self.k = K
        self.use_avg_pool = args.use_avg_pool

        self.input_transform_net = transform_net(args, 6, 3)

        self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64 * 2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64 * 2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128 * 2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        if self.use_avg_pool:
            # use avepooling + maxpooling
            self.conv5 = nn.Conv1d(num_f_prev, 512, kernel_size=1, bias=False)
            self.bn5 = nn.BatchNorm1d(512)
        else:
            # use only maxpooling
            self.conv5 = nn.Conv1d(num_f_prev, 1024, kernel_size=1, bias=False)
            self.bn5 = nn.BatchNorm1d(1024)
        
        self.cls = linear_classifier(1024, args.num_class)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        cls_logits = {}

        x = get_graph_feature(x, self.args, k=self.k)  # x: [b, 6, 1024, 20]
        x = self.conv1(x)  # x: [b, 64, 1024, 20]
        x1 = x.max(dim=-1, keepdim=False)[0]            # B, 64, 1024

        x = get_graph_feature(x1, self.args, k=self.k)      # [b, 128, 1024, 20]
        x = self.conv2(x)                                   # [b, 64, 1024, 20]
        x2 = x.max(dim=-1, keepdim=False)[0]                # [b, 64, 1024]

        x = get_graph_feature(x2, self.args, k=self.k)      # [b, 128, 1024, 20]
        x = self.conv3(x)                                   # [b, 128, 1024, 20]
        x3 = x.max(dim=-1, keepdim=False)[0]                # [b, 128, 1024]

        x = get_graph_feature(x3, self.args, k=self.k)      # [b, 256, 1024, 20]
        x = self.conv4(x)                                   # [b, 256, 1024, 20]
        x4 = x.max(dim=-1, keepdim=False)[0]                # [b, 256, 1024]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)          # [b, 512, 1024]

        if self.use_avg_pool:
            x5 = self.conv5(x_cat)                          # [b, 512, 1024]
            x5 = F.leaky_relu(self.bn5(x5), negative_slope=0.2)
            x5_1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
            x5_2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)
            x5_pool = torch.cat((x5_1, x5_2), 1)
        else:
            x5 = self.conv5(x_cat)                          # [b, 512, 1024]
            x5 = F.leaky_relu(self.bn5(x5), negative_slope=0.2)
            x5_pool = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)

        x = x5_pool

        cls_logits['feature'] = x

        cls_logits['pred'] = self.cls(x)
        
        return cls_logits
    

class segmentation(nn.Module):
    def __init__(self, args, input_size, output_size):
        super(segmentation, self).__init__()
        self.args = args
        self.of1 = 256
        self.of2 = 256
        self.of3 = output_size

        self.bn1 = nn.BatchNorm1d(self.of1)
        self.bn2 = nn.BatchNorm1d(self.of2)
        self.bn3 = nn.BatchNorm1d(self.of3)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.dp2 = nn.Dropout(p=args.dropout)

        self.conv1 = nn.Conv1d(input_size, self.of1, kernel_size=1, bias=True)
        self.conv2 = nn.Conv1d(self.of1, self.of2, kernel_size=1, bias=True)
        self.conv3 = nn.Conv1d(self.of2, self.of3, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.dp1(F.relu(self.bn1(self.conv1(x))))
        x = self.dp2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        return x.permute(0, 2, 1)                           # [b, 1024, 128]
    

class linear_DGCNN_seg_model(nn.Module):
    def __init__(self, args):
        super(linear_DGCNN_seg_model, self).__init__()
        self.args = args
        self.k = K
        self.use_avg_pool = args.use_avg_pool

        self.input_transform_net = transform_net(args, 6, 3)

        self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64 * 2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64 * 2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128 * 2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        if self.use_avg_pool:
            # use avepooling + maxpooling
            self.conv5 = nn.Conv1d(num_f_prev, 512, kernel_size=1, bias=False)
            self.bn5 = nn.BatchNorm1d(512)
        else:
            # use only maxpooling
            self.conv5 = nn.Conv1d(num_f_prev, 1024, kernel_size=1, bias=False)
            self.bn5 = nn.BatchNorm1d(1024)
        
        self.seg = segmentation(args, input_size=1024 + 512, output_size=args.feature_dim)            
        self.seg_cls = linear_classifier(args.feature_dim, args.num_class)           # default: 8

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        cls_logits = {}

        x = get_graph_feature(x, self.args, k=self.k)  # x: [b, 6, 1024, 20]
        x = self.conv1(x)  # x: [b, 64, 1024, 20]
        x1 = x.max(dim=-1, keepdim=False)[0]            # B, 64, 1024

        x = get_graph_feature(x1, self.args, k=self.k)      # [b, 128, 1024, 20]
        x = self.conv2(x)                                   # [b, 64, 1024, 20]
        x2 = x.max(dim=-1, keepdim=False)[0]                # [b, 64, 1024]

        x = get_graph_feature(x2, self.args, k=self.k)      # [b, 128, 1024, 20]
        x = self.conv3(x)                                   # [b, 128, 1024, 20]
        x3 = x.max(dim=-1, keepdim=False)[0]                # [b, 128, 1024]

        x = get_graph_feature(x3, self.args, k=self.k)      # [b, 256, 1024, 20]
        x = self.conv4(x)                                   # [b, 256, 1024, 20]
        x4 = x.max(dim=-1, keepdim=False)[0]                # [b, 256, 1024]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)          # [b, 512, 1024]

        if self.use_avg_pool:
            x5 = self.conv5(x_cat)                          # [b, 512, 1024]
            x5 = F.leaky_relu(self.bn5(x5), negative_slope=0.2)
            x5_1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
            x5_2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)
            x5_pool = torch.cat((x5_1, x5_2), 1)
        else:
            x5 = self.conv5(x_cat)                          # [b, 1024, 1024]
            x5 = F.leaky_relu(self.bn5(x5), negative_slope=0.2)
            x5_pool = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)

        x = torch.cat((x_cat, x5_pool.unsqueeze(2).repeat(1, 1, num_points)), dim=1)        # [b, 1536, 1024]

        seg_feature = self.seg(x)           # [b, 1024, 128]
        seg_pred = self.seg_cls(seg_feature)

        cls_logits['feature'] = seg_feature

        cls_logits['pred'] = seg_pred
        
        return cls_logits
    

class class_classifier(nn.Module):
    def __init__(self, args, input_dim, num_class=10):
        super(class_classifier, self).__init__()

        activate = 'leakyrelu' if args.model == 'DGCNN' else 'relu'
        bias = True if args.model == 'DGCNN' else False

        self.mlp1 = fc_layer(input_dim, 512, bias=bias, activation=activate, bn=True)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.mlp2 = fc_layer(512, 256, bias=True, activation=activate, bn=True)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.dp1(self.mlp1(x))
        x = self.dp2(self.mlp2(x))
        x = self.mlp3(x)
        return x


class linear_classifier(nn.Module):
    def __init__(self, input_dim, num_class):
        super(linear_classifier, self).__init__()

        self.mlp = nn.Linear(input_dim, num_class)

    def forward(self, x):
        x = self.mlp(x)
        return x
    

if __name__ == '__main__':

    def str2bool(v):
        '''
        Input:
            v - string
        output:
            True/False
        '''
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='DA on Point Clouds')
    parser.add_argument('--dataroot', type=str, default='../gast/data/', metavar='N', help='data path')
    parser.add_argument('--model', type=str, default='DGCNN', choices=['pointnet', 'DGCNN'], help='Model to use')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='1',
                        help='comma delimited of gpu ids to use. Use -1 for cpu usage')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes per dataset')
    parser.add_argument('--use_avg_pool', type=str2bool, default=False, help='Using average pooling & max pooling or max pooling only')
    parser.add_argument('--batch_size', type=int, default=20, metavar='batch_size',
                        help='Size of train batch per domain')
    parser.add_argument('--test_batch_size', type=int, default=20, metavar='batch_size',
                        help='Size of test batch per domain')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='threshold for pseudo label')
    parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
    parser.add_argument('--exp_name', type=str, default='test', help='Name of the experiment')

    args = parser.parse_args()

    args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()

    data = torch.rand(4, 3, 1024).cuda()
    model = DGCNN_model(args).cuda()
    out = model(data)
    # print(1)

