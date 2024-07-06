import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import gc
import math
from sklearn import manifold


class EstimatorCV():
    def __init__(self, feature_num, class_num, device):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.device = device

        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).to(device)
        self.Ave = torch.zeros(class_num, feature_num).to(device)
        self.Amount = torch.zeros(class_num).to(device)

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).to(self.device)
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)       # feature of a certain class

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                      .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        self.Amount += onehot.sum(0)


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num, device):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num, device)

        self.class_num = class_num
        self.device = device

        self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        # sigma2 = ratio * \
        #          torch.bmm(torch.bmm(NxW_ij - NxW_kj,
        #                              CV_temp).view(N * C, 1, A),
        #                    (NxW_ij - NxW_kj).view(N * C, A, 1)).view(N, C)

        sigma2 = ratio * \
                 torch.bmm(torch.bmm(NxW_ij - NxW_kj,
                                     CV_temp),
                           (NxW_ij - NxW_kj).permute(0, 2, 1))

        sigma2 = sigma2.mul(torch.eye(C).to(self.device)
                            .expand(N, C, C)).sum(2).view(N, C)

        aug_result = y + 0.5 * sigma2

        return aug_result

    def forward(self, model, fc, x, target_x, ratio):

        logits = model(x)
        features = logits['feature']
        y = logits['pred']

        self.estimator.update_CV(features.detach(), target_x)

        isda_aug_y = self.isda_aug(fc, features, y, target_x, self.estimator.CoVariance.detach(), ratio)

        loss = self.cross_entropy(isda_aug_y, target_x)

        return loss, y
    

def MI(outputs_target):
    batch_size = outputs_target.size(0)
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size)
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t)) / float(batch_size)
    return item1 - item2

def CalculateMean(features, labels, class_num):
    device = features.device
    N = features.size(0)        # size of the pool
    C = class_num
    A = features.size(1)        # dimension of the feature

    avg_CxA = torch.zeros(C, A).to(device)
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).to(device)
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1.0

    del onehot
    gc.collect()
    for c in range(class_num):
        c_temp = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        c_temp = torch.sum(c_temp, dim=0)
        avg_CxA[c] = c_temp / Amount_CxA[c]
    return avg_CxA.detach()

def Calculate_CV(features, labels, ave_CxA, class_num):
    device = features.device
    N = features.size(0)
    C = class_num
    A = features.size(1)

    var_temp = torch.zeros(C, A, A).to(device)
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).to(device)
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1
    Amount_CxAxA = Amount_CxA.view(C, A, 1).expand(C, A, A)
    del Amount_CxA, onehot
    gc.collect()

    avg_NxCxA = ave_CxA.expand(N, C, A)
    for c in range(C):
        features_by_sort_c = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        avg_by_sort_c = avg_NxCxA[:, c, :].mul(NxCxA_onehot[:, c, :])
        var_temp_c = features_by_sort_c - avg_by_sort_c
        var_temp[c] = torch.mm(var_temp_c.permute(1,0), var_temp_c).div(Amount_CxAxA[c])
    return var_temp.detach()


class TSALoss(nn.Module):
    def __init__(self, class_num):
        super(TSALoss, self).__init__()
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

    def aug(self, s_mean_matrix, t_mean_matrix, fc, features, y_s, labels_s, t_cv_matrix, Lambda):
        device = features.device
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels_s.view(N, 1, 1).expand(N, C, A))

        t_CV_temp = t_cv_matrix[labels_s]

        sigma2 = Lambda * torch.bmm(torch.bmm(NxW_ij - NxW_kj, t_CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).to(device).expand(N, C, C)).sum(2).view(N, C)

        sourceMean_NxA = s_mean_matrix[labels_s]
        targetMean_NxA = t_mean_matrix[labels_s]
        dataMean_NxA = (targetMean_NxA - sourceMean_NxA)
        dataMean_NxAx1 = dataMean_NxA.expand(1, N, A).permute(1, 2, 0)

        del t_CV_temp, sourceMean_NxA, targetMean_NxA, dataMean_NxA
        gc.collect()

        dataW_NxCxA = NxW_ij - NxW_kj
        dataW_x_detaMean_NxCx1 = torch.bmm(dataW_NxCxA, dataMean_NxAx1)
        datW_x_detaMean_NxC = dataW_x_detaMean_NxCx1.view(N, C)

        aug_result = y_s + 0.5 * sigma2 + Lambda * datW_x_detaMean_NxC
        return aug_result

    def forward(self, fc, features_source: torch.Tensor, y_s, labels_source, Lambda, mean_source, mean_target, covariance_target):
        aug_y = self.aug(mean_source, mean_target, fc, features_source, y_s, labels_source, covariance_target, Lambda)
        loss = self.cross_entropy(aug_y, labels_source)
        return loss
    

def CalculateSelectedMean(feature, label, indicator, class_num):
    # feature: feature pool [N, A]
    # label: label pool [N]
    # indicator: indicate whether the sample is selected or not [N]
    device = feature.device

    N = feature.size(0)        # size of the pool, s_len + t_len
    C = class_num
    A = feature.size(1)        # dimension of the feature

    avg_CxA = torch.zeros(C, A).to(device)

    NxCxFeatures = feature.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).to(device)
    onehot.scatter_(1, label.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)     # if class_i = j, [i, :, j] = 1, else 0
    NxCxA_onehot = NxCxA_onehot * indicator.view(N, 1, 1).expand(N, C, A)       # incase some samples are not selected and would be treated as label = 0

    Amount_CxA = NxCxA_onehot.sum(0)        # [C, A], we need to calculate the mean for each channel for each category
    Amount_CxA[Amount_CxA == 0] = 1.0

    del onehot
    gc.collect()

    for c in range(class_num):
        c_temp = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        c_temp = torch.sum(c_temp, dim=0)
        avg_CxA[c] = c_temp / Amount_CxA[c]
    return avg_CxA.detach()


def CalculateSelectedCV(feature, label, indicator, mean_pool, class_num):
    # feature: feature pool [N, A]
    # label: label pool [N]
    # indicator: indicate whether the sample is selected or not [N]
    # mean_pool: mean value of the pool [C, A]
    device = feature.device

    N = feature.size(0)
    C = class_num
    A = feature.size(1)

    var_temp = torch.zeros(C, A, A).to(device)

    NxCxFeatures = feature.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).to(device)
    onehot.scatter_(1, label.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
    NxCxA_onehot = NxCxA_onehot * indicator.view(N, 1, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1
    Amount_CxAxA = Amount_CxA.view(C, A, 1).expand(C, A, A)
    
    del Amount_CxA, onehot
    gc.collect()

    avg_NxCxA = mean_pool.expand(N, C, A)
    for c in range(C):
        features_by_sort_c = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        avg_by_sort_c = avg_NxCxA[:, c, :].mul(NxCxA_onehot[:, c, :])
        var_temp_c = features_by_sort_c - avg_by_sort_c
        var_temp[c] = torch.mm(var_temp_c.permute(1,0), var_temp_c).div(Amount_CxAxA[c])
    return var_temp.detach()


class PCFEALoss_no_mean(nn.Module):
    def __init__(self, class_num):
        super(PCFEALoss_no_mean, self).__init__()
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

    def aug(self, fc, features, pred, labels, cv_matrix, Lambda):
        device = features.device

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))

        CV_temp = cv_matrix[labels]

        sigma2 = Lambda * torch.bmm(torch.bmm(NxW_ij - NxW_kj, CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).to(device).expand(N, C, C)).sum(2).view(N, C)

        aug_result = pred + 0.5 * sigma2

        return aug_result

    def forward(self, fc, features, pred, labels, Lambda, covariance_sample):
        aug_y = self.aug(fc, features, pred, labels, covariance_sample, Lambda)
        loss = self.cross_entropy(aug_y, labels)
        return loss
    

class PCFEALoss(nn.Module):
    def __init__(self, class_num):
        super(PCFEALoss, self).__init__()
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

    def aug(self, mean_matrix, mean_source, fc, features, pred, labels, cv_matrix, Lambda):
        device = features.device

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))

        CV_temp = cv_matrix[labels]

        sigma2 = Lambda * torch.bmm(torch.bmm(NxW_ij - NxW_kj, CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).to(device).expand(N, C, C)).sum(2).view(N, C)

        sourceMean_NxA = mean_source[labels]
        poolMean_NxA = mean_matrix[labels]
        dataMean_NxA = (poolMean_NxA - sourceMean_NxA)
        dataMean_NxAx1 = dataMean_NxA.expand(1, N, A).permute(1, 2, 0)

        dataW_NxCxA = NxW_ij - NxW_kj
        dataW_x_detaMean_NxCx1 = torch.bmm(dataW_NxCxA, dataMean_NxAx1)
        datW_x_detaMean_NxC = dataW_x_detaMean_NxCx1.view(N, C)

        aug_result = pred + 0.5 * sigma2 + datW_x_detaMean_NxC

        return aug_result

    def forward(self, fc, features, pred, labels, Lambda, mean_sample, mean_source, covariance_sample):
        aug_y = self.aug(mean_sample, mean_source, fc, features, pred, labels, covariance_sample, Lambda)
        loss = self.cross_entropy(aug_y, labels)
        return loss
    

class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(Focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, label):
        device = pred.device
        B, D = pred.shape                   # D is num_class
        # pred = torch.sigmoid(pred)          # which function can be used here
        pred = F.softmax(pred, dim=-1)
        ones = torch.sparse.torch.eye(D).to(device)
        label_one_hot = ones.index_select(0, label)
        pred = (pred * label_one_hot).sum(dim=-1)
        loss = -self.alpha * (torch.pow((1 - pred), self.gamma)) * pred.log()

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            loss = loss

        return loss


def dot(x, y):
    return torch.sum(x * y, dim=-1)


# SimCLR
class InfoNCE(nn.Module):
    def __init__(self, temperature):
        super(InfoNCE, self).__init__()
        self.T = temperature
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, data1, data2):
        data = torch.cat([data1, data2], dim=0)     # 2*B, D
        device = data1.device
        B, D = data1.shape
        sim = self.cossim(data.unsqueeze(0), data.unsqueeze(1)) / self.T
        sim_pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0).reshape(2*B, 1)
        mask = torch.ones_like(sim).long().to(device)
        idx1 = torch.ones([B]).long().to(device)
        idx2 = torch.ones([2*B]).long().to(device)
        mask = mask - (torch.diag_embed(idx1, B) + torch.diag_embed(idx1, -B) + torch.diag_embed(idx2, 0))
        mask = mask.bool()
        sim_neg = sim[mask].reshape(2*B, -1)

        sim = torch.cat([sim_pos, sim_neg], dim=-1)
        label = torch.zeros(2*B).long().to(device)
        loss = self.CELoss(sim, label)
        return loss


def IDFALoss(prototype, feature, label, tao):
    # adopted from PCS
    # prototype: num_class, D
    # feature: batch_size, D
    proto_feature = prototype[label]
    sim_pos = torch.exp(torch.cosine_similarity(feature, proto_feature, dim=-1) / tao)
    sim_neg = torch.exp(torch.cosine_similarity(feature.unsqueeze(1), prototype.unsqueeze(0), dim=-1) / tao).sum(dim=-1)
    # sim_pos = torch.exp(torch.sum(feature * proto_feature, dim=-1) / tao)
    # sim_neg = torch.sum(torch.exp(torch.mm(feature, prototype.T) / tao), dim=-1)
    ratio = sim_pos / (sim_neg + 1e-6)
    proto_loss = -1 * torch.sum(torch.log(ratio)) / (ratio.size(0) + 1e-6)
    
    return proto_loss


def sigmoid_function(input, k):
    return 1.0/(1 + math.exp(-input * k))


# calculate similarity between samples and center of clusters in PCS
# this function is in torch utils in utils folder in pcs folder
def contrastive_sim(instances, proto=None, tao=0.05):
    # prob_matrix [bs, dim]
    # proto_dim [nums, dim]
    if proto is None:
        proto = instances
    ins_ext = instances.unsqueeze(1).repeat(1, proto.size(0), 1)
    sim_matrix = torch.exp(torch.sum(ins_ext * proto, dim=-1) / tao)
    return sim_matrix


def cosine_similarity(data, center):
    data = data.view(data.shape[0], -1)
    center = center.view(center.shape[0], -1)
    data = F.normalize(data)
    center = F.normalize(center)
    distance = data.mm(center.t())

    return distance


if __name__ == '__main__':
    data1 = torch.rand([24, 3, 1024]).cuda()
    data2 = torch.rand([24, 3, 512]).cuda()
    