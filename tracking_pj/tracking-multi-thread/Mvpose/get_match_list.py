import torch
import os
import os.path as osp
import numpy as np
import time


def transform_closure(match_list):
    """
    eliminate null items of match list.
    """
    for i in range(len(match_list) - 1, -1, -1):
        if match_list[i].shape[0] == 0:
            del (match_list[i])

    return match_list


def myproj2dpam(Y, tol=1e-4):
    X0 = Y
    X = Y
    I2 = 0

    for iter_ in range(10):

        X1 = projR(X0 + I2)
        I1 = X1 - (X0 + I2)
        X2 = projC(X0 + I1)
        I2 = X2 - (X0 + I1)

        chg = torch.sum(torch.abs(X2[:] - X[:])) / X.numel()
        X = X2
        if chg < tol:
            return X
    return X


def matchSVT(S, dimGroup, **kwargs):
    alpha = kwargs.get('alpha', 0.1)
    pSelect = kwargs.get('pselect', 1)
    tol = kwargs.get('tol', 5e-4)
    maxIter = kwargs.get('maxIter', 500)
    _lambda = kwargs.get('_lambda', 50)
    mu = kwargs.get('mu', 64)
    dual_stochastic = kwargs.get('dual_stochastic_SVT', True)

    N = S.shape[0]
    S[torch.arange(N), torch.arange(N)] = 0
    S = (S + S.t()) / 2
    X = S.clone()
    Y = torch.zeros_like(S)
    W = alpha - S
    t0 = time.time()

    for iter_ in range(maxIter):
        X0 = X
        # update Q with SVT
        U, s, V = torch.svd(1.0 / mu * Y + X)  # Singular Value Decomposition，奇异值分解
        diagS = s - _lambda / mu
        diagS[diagS < 0] = 0
        Q = U @ diagS.diag() @ V.t()
        # update X
        X = Q - (W + Y) / mu
        # project X
        for i in range(len(dimGroup) - 1):
            ind1, ind2 = dimGroup[i], dimGroup[i + 1]
            X[ind1:ind2, ind1:ind2] = 0
        if pSelect == 1:
            X[torch.arange(N), torch.arange(N)] = 1
        X[X < 0] = 0
        X[X > 1] = 1

        if dual_stochastic:
            # Projection for double stochastic constraint
            for i in range(len(dimGroup) - 1):
                row_begin, row_end = int(dimGroup[i]), int(dimGroup[i + 1])
                for j in range(len(dimGroup) - 1):
                    col_begin, col_end = int(dimGroup[j]), int(dimGroup[j + 1])
                    if row_end > row_begin and col_end > col_begin:
                        X[row_begin:row_end, col_begin:col_end] = myproj2dpam(X[row_begin:row_end, col_begin:col_end],
                                                                              1e-2)

        X = (X + X.t()) / 2
        # update Y
        Y = Y + mu * (X - Q)
        # test if convergence
        pRes = torch.norm(X - Q) / N
        dRes = mu * torch.norm(X - X0) / N

        if pRes < tol and dRes < tol:
            break

        if pRes > 10 * dRes:
            mu = 2 * mu
        elif dRes > 10 * pRes:
            mu = mu / 2

    X = (X + X.t()) / 2

    X_bin = X > 0.5
    match_mat = X_bin.numpy()

    return torch.tensor(match_mat)


def get_match_list(reid_affinity_mat, geo_affinity_mat=None, dimGroup=None):
    # fuse two mats with multiplay
    # W = torch.sqrt(reid_affinity_mat * geo_affinity_mat)

    W = reid_affinity_mat
    W[torch.isnan(W)] = 0  # Some times (Shelf 452th img eg.) torch.sqrt will return nan if its too small

    num_person = 10
    X0 = torch.rand(W.shape[0], num_person)

    # Use spectral method to initialize assignment matrix.
    eig_value, eig_vector = W.eig(eigenvectors=True)  # 求特征值、特征向量
    _, eig_idx = torch.sort(eig_value[:, 0], descending=True)

    if W.shape[1] >= num_person:
        X0 = eig_vector[eig_idx[:num_person]].t()
    else:
        X0[:, :W.shape[1]] = eig_vector.t()

    match_mat = matchSVT(W, dimGroup, alpha=0.5, _lambda=50, dual_stochastic_SVT=False)  # cfg配置文件

    # ★由匹配矩阵生成匹配列表
    bin_match = match_mat[:, torch.nonzero(torch.sum(match_mat, dim=0) > 1.9).squeeze()] > 0.9
    bin_match = bin_match.reshape(W.shape[0], -1)
    try:
        matched_list = [[] for i in range(bin_match.shape[1])]
    except:
        print('bin_match')
        print(bin_match)
        return []
    for sub_imgid, row in enumerate(bin_match):
        if row.sum() != 0:
            pid = row.argmax()
            matched_list[pid].append(sub_imgid)
    matched_list = [np.array(i) for i in matched_list]

    matched_list = transform_closure(matched_list)  # remove null items in list
    return matched_list


if __name__ == '__main__':
    dimGroup = [0, 3, 5, 7]  # tag of RoIs: 0,1,2 from cam_1 ; 3,4 from cam_2 ; 5,6 from cam_3

    reid_affinity_mat = torch.tensor([
        [9.9992e-01, 1.7375e-02, 3.7205e-03, 8.6547e-01, 1.7083e-01, 6.6996e-02, 9.8977e-01],
        [1.7375e-02, 9.9992e-01, 6.4126e-01, 9.4190e-02, 5.8224e-02, 5.6361e-02, 7.5706e-02],
        [3.7205e-03, 6.4126e-01, 9.9992e-01, 2.6453e-03, 9.4926e-04, 4.2990e-03, 6.4881e-03],
        [8.6547e-01, 9.4190e-02, 2.6453e-03, 9.9992e-01, 8.1866e-01, 2.4143e-01, 9.0384e-01],
        [1.7083e-01, 5.8224e-02, 9.4926e-04, 8.1866e-01, 9.9992e-01, 9.9592e-01, 3.5098e-01],
        [6.6996e-02, 5.6361e-02, 4.2990e-03, 2.4143e-01, 9.9592e-01, 9.9992e-01, 4.8742e-01],
        [9.8977e-01, 7.5706e-02, 6.4881e-03, 9.0384e-01, 3.5098e-01, 4.8742e-01, 9.9992e-01]])

    geo_affinity_mat = torch.tensor([
        [9.8263e-01, 3.5740e-08, 7.6009e-04, 9.7411e-01, 8.9836e-01, 2.3110e-01, 9.7667e-01],
        [3.5740e-08, 9.8263e-01, 5.9793e-04, 9.5251e-01, 9.5662e-01, 1.7477e-02, 6.5769e-01],
        [7.6009e-04, 5.9793e-04, 9.8263e-01, 8.8219e-01, 9.5404e-01, 9.1456e-03, 5.7129e-01],
        [9.7411e-01, 9.5251e-01, 8.8219e-01, 9.8263e-01, 3.3865e-01, 8.4012e-01, 9.6965e-01],
        [8.9836e-01, 9.5662e-01, 9.5404e-01, 3.3865e-01, 9.8263e-01, 9.6444e-01, 9.2377e-01],
        [2.3110e-01, 1.7477e-02, 9.1456e-03, 8.4012e-01, 9.6444e-01, 9.8263e-01, 1.3049e-02],
        [9.7667e-01, 6.5769e-01, 5.7129e-01, 9.6965e-01, 9.2377e-01, 1.3049e-02, 9.8263e-01]])

    match_list = get_match_list(reid_affinity_mat, geo_affinity_mat, dimGroup)  # match_list with null items

    print(match_list)  # ex: [array([0, 3, 6]), array([4, 5])]
    #  witch means that 0,3,6 RoIs are the same person, and 4,5 RoIs are another person
