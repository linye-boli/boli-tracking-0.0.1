import torch
import numpy as np


# ### Re-ID functions
def pairwise_affinity(query_features, gallery_features):
    # 成对地使用欧式距离进行比较  import pdb; pdb.set_trace()

    # ★构建f*f大小的矩阵
    x = torch.cat([query_features[i].unsqueeze(0) for i in range(0, len(query_features))], 0)
    y = torch.cat([gallery_features[j].unsqueeze(0) for j in range(0, len(gallery_features))], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    # ★计算欧式距离，并对矩阵进行归一化，使用sigmoid进行映射
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    normalized_affinity = - (dist - dist.mean()) / dist.std()
    affinity = torch.sigmoid(normalized_affinity * torch.tensor(5.))  # x5 to match 1->1

    return affinity


def get_reid_affinity(query_features):
    """
    Will compute matrix with itself
    query_features：特征：Re-ID features
    """
    # ★得到归一化后的A矩阵
    affinity = pairwise_affinity(query_features, query_features.copy())

    return affinity


# ### Geo functions
def projected_distance(pts_0, pts_1, F):
    """
    Compute point distance with epipolar geometry knowledge
    :param pts_0: numpy points array with shape Nx17x2
    :param pts_1: numpy points array with shape Nx17x2
    :param F: Fundamental matrix F_{01}
    :return: numpy array of pairwise distance
    """

    lines = cv2.computeCorrespondEpilines(pts_0.reshape(-1, 1, 2), 2, F)
    lines = lines.reshape(-1, 17, 1, 3)
    lines = lines.transpose(0, 2, 1, 3)
    points_1 = np.ones([1, pts_1.shape[0], 17, 3])
    points_1[0, :, :, :2] = pts_1

    dist = np.sum(lines * points_1, axis=3)  # / np.linalg.norm(lines[:, :, :, :2], axis=3)
    dist = np.abs(dist)
    dist = np.mean(dist, axis=2)

    return dist


def geometry_affinity(pose_mat, Fs, dimGroup):
    """
    :param pose_mat:共有M个人，维度M*17*3
    :param Fs:对极几何中将一像平面的点映射到另一像平面的线，阶为2的3*3矩阵
    :param dimGroup: 用于RoI分组，存储RoI和Cam的信息。[0,3,5,8]，左开右闭原则(012)(34)(567)没有8
    :return:
    """
    M, _, _ = pose_mat.shape  # 共有M个人
    distance_matrix = np.ones((M, M), dtype=np.float32) * 25
    np.fill_diagonal(distance_matrix, 0)  # 对角元素设置为0

    # TODO: remove this stupid nested for loop
    # 一个镜头为一个组，平面铺开，互不重叠地进行对比
    for i in range(len(dimGroup) - 1):
        for j in range(i, len(dimGroup) - 1):
            pose_id0 = pose_mat[dimGroup[i]:dimGroup[i + 1]]
            pose_id1 = pose_mat[dimGroup[j]:dimGroup[j + 1]]

            if len(pose_id0) == 0 or len(pose_id1) == 0:
                continue

            distance_matrix[dimGroup[i]:dimGroup[i + 1], dimGroup[j]:dimGroup[j + 1]] = \
                (projected_distance(pose_id0, pose_id1, Fs[i, j]) + projected_distance(pose_id1, pose_id0,
                                                                                       Fs[j, i]).T) / 2

            distance_matrix[dimGroup[j]:dimGroup[j + 1], dimGroup[i]:dimGroup[i + 1]] = \
                distance_matrix[dimGroup[i]:dimGroup[i + 1], dimGroup[j]:dimGroup[j + 1]].T

    # 归一化
    geo_affinity_matrix = - (distance_matrix - distance_matrix.mean()) / distance_matrix.std()
    # TODO: add flexible factor
    geo_affinity_matrix = 1 / (1 + np.exp(-5 * geo_affinity_matrix))
    return geo_affinity_matrix
