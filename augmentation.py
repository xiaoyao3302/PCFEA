import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import open3d as o3d
import pdb
# from models.pointnet2_utils import farthest_point_sample, query_ball_point, index_points


def reshape_num(data, num_point):
    """

    :param data: N * 3
    :param num_point:
    :return:
    """
    len_data = data.shape[0]
    if len_data < num_point:
        num_cat = math.ceil(num_point / len_data)
        data = data.repeat(num_cat, 0)[:num_point, :]
    else:
        data = data[:num_point, :]

    return data


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_with_normal(batch_xyz_normal):
    """ Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    """
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k, :, 0:3]
        shape_normal = batch_xyz_normal[k, :, 3:6]
        batch_xyz_normal[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal


def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """ batch_pc: BxNx3 """
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


def crop_point(data):
    """
    # random crop: along one axis
    :param data: B, N, C    0 - 1
    :return:
    """
    B, N, C = data.shape
    data_new = []
    for ii in range(B):
        len_x = data[ii, :, 0].max() - data[ii, :, 0].min()
        threshold = data[ii, :, 0].min() + len_x * (np.random.random() * 0.2 + 0.4)      # 0.4 - 0.6
        data1_indicate = data[ii, :, 0] < threshold
        data2_indicate = data[ii, :, 0] >= threshold

        num_data1 = data1_indicate.sum()
        num_data2 = data2_indicate.sum()

        data_indicate = data1_indicate if num_data1 > num_data2 else data2_indicate
        data_select = data[ii, data_indicate, :]

        data_select = reshape_num(data_select, num_point=1024)

        if len(data_new) == 0:
            data_new = np.expand_dims(data_select, axis=0)
        else:
            data_new = np.concatenate([data_new, np.expand_dims(data_select, axis=0)], axis=0)

    return data_new


def rotate_point_cloud_3d(pc):
    rotation_angle = np.random.rand(3) * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix_1 = np.array([[cosval[0], 0, sinval[0]],
                                  [0, 1, 0],
                                  [-sinval[0], 0, cosval[0]]])
    rotation_matrix_2 = np.array([[1, 0, 0],
                                  [0, cosval[1], -sinval[1]],
                                  [0, sinval[1], cosval[1]]])
    rotation_matrix_3 = np.array([[cosval[2], -sinval[2], 0],
                                  [sinval[2], cosval[2], 0],
                                  [0, 0, 1]])
    rotation_matrix = np.matmul(np.matmul(rotation_matrix_1, rotation_matrix_2), rotation_matrix_3)
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def density(pc, num_point=1024):
    # N, C
    try:
        rand_points = np.random.uniform(-1, 1, 40000)
        x1 = rand_points[:20000]
        x2 = rand_points[20000:]
        power_sum = x1 ** 2 + x2 ** 2
        p_filter = power_sum < 1
        power_sum = power_sum[p_filter]
        sqrt_sum = np.sqrt(1 - power_sum)
        x1 = x1[p_filter]
        x2 = x2[p_filter]
        x = (2 * x1 * sqrt_sum).reshape(-1, 1)
        y = (2 * x2 * sqrt_sum).reshape(-1, 1)
        z = (1 - 2 * power_sum).reshape(-1, 1)
        density_points = np.hstack([x, y, z])
        v_point = density_points[np.random.choice(density_points.shape[0])]

        gate = np.random.uniform(low=1.3, high=1.6)
        dist = np.sqrt((v_point ** 2).sum())
        max_dist = dist + 1
        min_dist = dist - 1
        dist = np.linalg.norm(pc - v_point.reshape(1, 3), axis=1)
        dist = (dist - min_dist) / (max_dist - min_dist)
        r_list = np.random.uniform(0.75, 1, pc.shape[0])
        tmp_pc = pc[dist * gate < (r_list)]

        num_pad = np.ceil(num_point / tmp_pc.shape[0]).astype(np.long)
        pc = np.tile(tmp_pc, (num_pad, 1))[:num_point]
    except:
        pc = pc
        
    return pc


def drop_hole(pc, num_point=1024):
    # N, C
    try:
        p = np.random.uniform(low=0.25, high=0.45)
        random_point = np.random.randint(0, pc.shape[0])
        index = np.linalg.norm(pc - pc[random_point].reshape(1, 3), axis=1).argsort()

        tmp_pc = pc[index[int(pc.shape[0] * p):]]
        num_pad = np.ceil(num_point / tmp_pc.shape[0]).astype(np.long)
        pc = np.tile(tmp_pc, (num_pad, 1))[:num_point]
    except:
        pc = pc

    return pc


def p_scan(pc, pixel_size=0.022, num_point=1024):
    # N, C
    try:
        pixel = int(2 / pixel_size)
        rotated_pc = rotate_point_cloud_3d(pc)
        pc_compress = (rotated_pc[:, 2] + 1) / 2 * pixel * pixel + (rotated_pc[:, 1] + 1) / 2 * pixel
        points_list = [None for i in range((pixel + 5) * (pixel + 5))]
        pc_compress = pc_compress.astype(np.int)
        for index, point in enumerate(rotated_pc):
            compress_index = pc_compress[index]
            if compress_index > len(points_list):
                print('out of index:', compress_index, len(points_list), point, pc[index], (pc[index] ** 2).sum(),
                        (point ** 2).sum())
            if points_list[compress_index] is None:
                points_list[compress_index] = index
            elif point[0] > rotated_pc[points_list[compress_index]][0]:
                points_list[compress_index] = index
        points_list = list(filter(lambda x: x is not None, points_list))
        points_list = pc[points_list]

        num_pad = np.ceil(num_point / points_list.shape[0]).astype(np.long)
        points_list = np.tile(points_list, (num_pad, 1))[:num_point]
    except:
        points_list = pc

    return points_list


def add_noise(data, noise=0.005):
    B, N, C = data.shape
    noise = (noise ** 0.5) * np.random.randn(B, N, C)
    data = data + noise
    return data


def weak_aug(data):
    # data: B, N, C
    device = data.device
    data = data.cpu().numpy()
    # data = random_point_dropout(data)
    data[:, :, 0:3] = random_scale_point_cloud(data[:, :, 0:3])
    data[:, :, 0:3] = shift_point_cloud(data[:, :, 0:3])
    return torch.Tensor(data).to(device)


def strong_aug(data, aug_type='2'):
    # data: B, N, C
    # v1
    device = data.device
    data = data.cpu().numpy()
    if aug_type == '1':
        data = normalize_data(data)
        # data = crop_point(data)
        data = density(data)
        # data = random_point_dropout(data)
        data = random_scale_point_cloud(data)
        data = shift_point_cloud(data)
        data = shuffle_points(data)
        data = rotate_point_cloud(data)
        data = add_noise(data)

    # v2
    else:
        data = normalize_data(data)
        # data = crop_point(data)
        data = drop_hole(data)
        # data = random_point_dropout(data)
        data = random_scale_point_cloud(data)
        data = shift_point_cloud(data)
        data = shuffle_points(data)
        data = rotate_point_cloud(data)
        data = add_noise(data)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(data[0, :, 0], data[0, :, 1], data[0, :, 2])

    return torch.Tensor(data).to(device)                 # B, N, C


# def mask_data(xyz, mask_p=0.5, npoint=64, radius=0.1, nsample=16):
#     if xyz.shape[1] < xyz.shape[2]:
#         xyz = xyz.transpose(2, 1)
#     B, N, C = xyz.shape
#     fps_idx = farthest_point_sample(xyz, npoint)        # 选出128个关键点
#     select_p = np.random.random_integers(0, npoint, [B, npoint])
#     mask_idx = torch.from_numpy(select_p > (mask_p * npoint))      # 以一定的概率随机选关键点，大于p的是留下来的
#     mask_idx = fps_idx * mask_idx
#     select_xyz = index_points(xyz, mask_idx)
#     idx = query_ball_point(radius, nsample, xyz, select_xyz)
#     grouped_xyz = index_points(xyz, idx).reshape(B, -1, C)

#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(grouped_xyz[3, :, 0], grouped_xyz[3, :, 1], grouped_xyz[3, :, 2])

#     return grouped_xyz


if __name__ == '__main__':
    dataroot = '../data/'
    DATA_DIR = os.path.join(dataroot, "PointDA_data", "shapenet_norm_curv_angle")
    npy_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', 'train', '*.npy')))
    pc_list = []
    for _dir in npy_list:
        pc_list.append(_dir)

    data = np.load(pc_list[18])[:, :3].astype(np.float32)

    data_raw = data
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data_raw)
    o3d.io.write_point_cloud('raw.ply', pcd)

    density_data = density(data)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(density_data)
    o3d.io.write_point_cloud('density.ply', pcd)

    drop_data = drop_hole(data)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(drop_data)
    o3d.io.write_point_cloud('drop.ply', pcd)

    scan_data = p_scan(data)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan_data)
    o3d.io.write_point_cloud('scan.ply', pcd)

    # pdb.set_trace()
    print(data.shape)

