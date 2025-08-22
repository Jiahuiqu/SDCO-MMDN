import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cv2
import math
from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift, random_walker
import torch
import os
from net1 import GraphOnly
import torch.nn as nn
from scipy.io import loadmat, savemat
import data_SLIC

from torch.autograd import Variable
seed = 0
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.

class MMD_loss(nn.Module):

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
                del XX, YY, XY, YX
            torch.cuda.empty_cache()
            return float(loss)


def LSC_superpixel(I, nseg):
    superpixelNum = nseg
    ratio = 0.075
    size = int(math.sqrt(((I.shape[0] * I.shape[1]) / nseg)))
    superpixelLSC = cv2.ximgproc.createSuperpixelLSC(
        I,
        region_size=size,
        ratio=0.005)
    superpixelLSC.iterate()
    superpixelLSC.enforceLabelConnectivity(min_element_size=25)
    segments = superpixelLSC.getLabels()
    return np.array(segments, np.int64)


def SEEDS_superpixel(I, nseg):
    I = np.array(I[:, :, 0:3], np.float32).copy()
    I_new = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    # I_new =np.array( I[:,:,0:3],np.float32).copy()
    height, width, channels = I_new.shape

    superpixelNum = nseg
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, int(superpixelNum), num_levels=2, prior=1,
                                               histogram_bins=5)
    seeds.iterate(I_new, 4)
    segments = seeds.getLabels()
    # segments=SegmentsLabelProcess(segments) # 排除labels中不连续的情况
    return segments


def SegmentsLabelProcess(labels):
    '''
    对labels做后处理，防止出现label不连续现象
    '''
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SLIC(object):
    def __init__(self, HSI, n_segments=1000, compactness=20, max_num_iter=20, sigma=0, min_size_factor=0.3,
                 max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_num_iter = max_num_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        # 数据standardization标准化,即提前全局BN
        height, width, bands = HSI.shape  # 原始高光谱数据的三个维度
        data = np.reshape(HSI, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])

    def get_Q_and_S_and_Segments(self):
        # 执行 SLIC 并得到Q(nxm),S(m*b)
        img = self.data
        (h, w, d) = img.shape
        # 计算超像素S以及相关系数矩阵Q

        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_num_iter=self.max_num_iter,
                        convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor,
                        slic_zero=False, start_label=1)

        # 判断超像素label是否连续,否则予以校正
        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = SegmentsLabelProcess(segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)

        # ######################################显示超像素图片
        out = mark_boundaries(img[:, :, [0, 1, 2]], segments)
        # out = (img[:, :, [0, 1, 2]]-np.min(img[:, :, [0, 1, 2]]))/(np.max(img[:, :, [0, 1, 2]])-np.min(img[:, :, [0, 1, 2]]))

        plt.figure()
        plt.imshow(out)  # 读取out，但不显示
        plt.show()  # 显示

        segments = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, d])

        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1

        self.S = S
        self.Q = Q

        return Q, S, self.segments

    def get_A(self, sigma: float):
        '''
         根据 segments 判定邻接矩阵
        :return:
        '''
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                # if len(sub_set)>1:
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue

                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A[idx1, idx2] = A[idx2, idx1] = diss

        return A

def get_superpixels(data, n_segments_init):
    # 超像素初始种子点个数
    n_segments_init = n_segments_init
    ls = data_SLIC.data_SLIC(data, 6)
    Q, S, A, Seg = ls.simple_superpixel(n_segments=n_segments_init)
    return Q, S, A, Seg

def compute_lossT(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):

    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we)/10
    return pool_cross_entropy

def compute_lossS(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):

    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we) / 300000
    return pool_cross_entropy


def GT_To_One_Hot(gt, h, w):
    '''
    Convet Gt to one-hot labels
    :param gt:
    :param class_count:
    :return:
    '''
    GT_One_Hot = []  # 转化为one-hot形式的标签
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(2, dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [h, w, 2])
    return GT_One_Hot


def main():
    criteron = nn.L1Loss()
    device = torch.device('cuda:1')
    epochs = 1800
    Cin = 224
    CN = 20
    N = 4
    lr = 0.001
    mmd = MMD_loss()
    root = "/run/media/xd132/E/ZTZ/CNMF_chage_detection/data/"
    T_n_segments_init = 500
    data_T1_tar = loadmat(os.path.join(root, "BAY/S1.mat" ))['HypeRvieW']
    data_T2_tar = loadmat(os.path.join(root, "BAY/S2.mat" ))['HypeRvieW']
    data_cat_tar = np.concatenate((data_T1_tar, data_T2_tar), axis=2)
    # data_cat_tar = data_T1_tar - data_T2_tar
    [Q_tar, _, _, _] = get_superpixels(data_cat_tar, T_n_segments_init)
    R_Q_tar = torch.from_numpy(Q_tar).to(device)
    Q_tar = R_Q_tar / (torch.sum(R_Q_tar, 0, keepdim=True))
    # print(Q_tar.shape)
    Table_tar = loadmat(os.path.join(root, "BAY/BAY5.mat"))['BAY']
    ht, wt, ct, = data_T1_tar.shape

    S_n_segments_init = 1200
    data_T1_src = loadmat(os.path.join(root, "BAR/Q1.mat" ))['HypeRvieW']
    data_T2_src = loadmat(os.path.join(root, "BAR/Q2.mat" ))['HypeRvieW']
    data_cat_src = np.concatenate((data_T1_src, data_T2_src), axis=2)
    # data_cat_src = data_T1_src - data_T2_src
    [Q_src, _, _, _] = get_superpixels(data_cat_src, S_n_segments_init)
    R_Q_src = torch.from_numpy(Q_src).to(device)
    Q_src = R_Q_src / (torch.sum(R_Q_src, 0, keepdim=True))
    Table_src = loadmat(os.path.join(root, "BAR/REF.mat"))['HypeRvieW']
    hs, ws, cs, = data_T1_src.shape


    # 获取训练样本的标签图
    S_gt = np.reshape(Table_src, [-1])
    S_gt = np.reshape(S_gt, [hs, ws])
    S_onehot = GT_To_One_Hot(S_gt, hs, ws)
    S_onehot = np.reshape(S_onehot, [-1, 2]).astype(int)

    T_gt = np.reshape(Table_tar, [-1])
    T_gt = np.reshape(T_gt, [ht, wt])
    T_onehot = GT_To_One_Hot(T_gt, ht, wt)
    T_onehot = np.reshape(T_onehot, [-1, 2]).astype(int)

    # ###########制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
    # 源域
    S_mask = np.zeros([hs * ws, 2])
    temp_ones = np.ones([2])
    S_gt = np.reshape(S_gt, [hs * ws])
    for i in range(hs * ws):
        if S_gt[i] != 0:
            S_mask[i] = temp_ones
    S_mask = np.reshape(S_mask, [hs * ws, 2])

    # 目标域
    T_mask = np.zeros([ht * wt, 2])
    temp_ones = np.ones([2])
    T_gt = np.reshape(T_gt, [ht * wt])
    for i in range(ht * wt):
        if T_gt[i] != 0:
            T_mask[i] = temp_ones
    T_mask = np.reshape(T_mask, [ht * wt, 2])

    # 转到GPU
    S_onehot = torch.from_numpy(S_onehot.astype(np.float32)).to(device)
    T_onehot = torch.from_numpy(T_onehot.astype(np.float32)).to(device)

    S_mask = torch.from_numpy(S_mask.astype(np.float32)).to(device)
    T_mask = torch.from_numpy(T_mask.astype(np.float32)).to(device)

    data_T1_src = torch.from_numpy(data_T1_src.astype(np.float32)).to(device)
    data_T2_src = torch.from_numpy(data_T2_src.astype(np.float32)).to(device)

    data_T1_tar = torch.from_numpy(data_T1_tar.astype(np.float32)).to(device)
    data_T2_tar = torch.from_numpy(data_T2_tar.astype(np.float32)).to(device)


    T1_flatten = data_T1_tar.reshape([ht * wt, -1])
    T2_flatten = data_T2_tar.reshape([ht * wt, -1])
    superpixels_flatten_T1 = torch.mm(Q_tar.t(), T1_flatten)
    superpixels_flatten_T2 = torch.mm(Q_tar.t(), T2_flatten)

    S1_flatten = data_T1_src.reshape([hs * ws, -1])
    S2_flatten = data_T2_src.reshape([hs * ws, -1])
    superpixels_flatten_S1 = torch.mm(Q_src.t(), S1_flatten)
    superpixels_flatten_S2 = torch.mm(Q_src.t(), S2_flatten)

    savemat('superpixels_flatten_T1.mat', {'x': superpixels_flatten_T1.detach().cpu().numpy()})
    savemat('superpixels_flatten_T2.mat', {'x': superpixels_flatten_T2.detach().cpu().numpy()})
    savemat('superpixels_flatten_S1.mat', {'x': superpixels_flatten_S1.detach().cpu().numpy()})
    savemat('superpixels_flatten_S2.mat', {'x': superpixels_flatten_S2.detach().cpu().numpy()})


    net = GraphOnly(Cin, CN, N).to(device)

    pretext_model = torch.load('Rstore/best.mdl')
    model2_dict = net.state_dict()
    state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    net.load_state_dict(model2_dict)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    best_loss = 99999
    # mmd = MMD_loss()
    net.train()
    for i in range(epochs):

        optimizer.zero_grad()
        GCN_result_T, GCN_result_S, RT, RS, R_T1, R_T2, R_S1, R_S2 = net(superpixels_flatten_T1, superpixels_flatten_T2, superpixels_flatten_S1, superpixels_flatten_S2)
        lossR_T = criteron(superpixels_flatten_T1, R_T1) + criteron(superpixels_flatten_T2, R_T2)
        lossR_S = criteron(superpixels_flatten_T1, R_T1) + criteron(superpixels_flatten_S2, R_S2)
        Super_RS = RS.argmax(dim = 1)
        Super_RT = RT.argmax(dim = 1)
        savemat('Super_RS.mat', {'x': Super_RS.detach().cpu().numpy()})
        savemat('Super_RT.mat', {'x': Super_RT.detach().cpu().numpy()})
        S_Change = []
        S_Unchange = []
        T_Change = []
        T_Unchange = []

        for p in range(len(Super_RT)):
            if Super_RT[p] == 0:
                T_Change.append(GCN_result_T[p].tolist())
            else:
                T_Unchange.append(GCN_result_T[p].tolist())
        for q in range(len(Super_RS)):
            if Super_RS[q] == 0:
                S_Change.append(GCN_result_S[q].tolist())
            else:
                S_Unchange.append(GCN_result_S[q].tolist())

        T_C = torch.tensor(T_Change)
        T_Uc = torch.tensor(T_Unchange)
        S_C = torch.tensor(S_Change)
        S_Uc = torch.tensor(S_Unchange)
        print(T_C.shape, S_C.shape)
        print(T_Uc.shape, S_Uc.shape)
        RS = torch.matmul(R_Q_src, RS)
        RT = torch.matmul(R_Q_tar, RT)
        loss_S = compute_lossS(RS, S_onehot, S_mask)
        loss_T = compute_lossT(RT, T_onehot, T_mask)
        if len(T_Unchange) != 1 and len(S_Unchange) != 0 and len(T_Change) != 0 and len(S_Change) != 0:
        # if len(T_Unchange) != 1 and len(S_Unchange) != 0 and len(T_Change) != 0 and len(S_Change) != 0 and len(
        #         T_Unchange) != 1 and len(S_Unchange) != 1 and len(T_Change) != 1 and len(S_Change) != 1:
            loss_mmd_Change = mmd(T_C, S_C)
            loss_mmd_Unchange = mmd(T_Uc, S_Uc)
            loss_mmd_T = torch.exp(torch.tensor(-mmd(T_C, T_Uc)))
            loss_mmd_S = torch.exp(torch.tensor(-mmd(S_C, S_Uc)))
            print("loss_mmd_Unchange:", loss_mmd_Unchange)
            print("loss_mmd_Change:", loss_mmd_Change)
            print('loss_mmd_T:', loss_mmd_T)
            print('loss_mmd_S:', loss_mmd_S)
            # loss = loss_S + loss_T + 0.001 * (loss_mmd_Unchange + loss_mmd_Change) + 0.1 * (loss_mmd_T+loss_mmd_S) + lossR_S + lossR_T
            loss = loss_S + loss_T + lossR_S + lossR_T
            print(i)
            print("loss_S:", loss_S)
            print("loss_T:", loss_T)
            print("lossR_T:", lossR_T)
            print("lossR_S:", lossR_S)
            print('loss:', loss)
        else:
            loss = 0.002 * loss_S + loss_T + lossR_S + lossR_T
        loss.backward(retain_graph=True)
        optimizer.step()

        if loss <= best_loss:
            print('更新best_loss，更新best.mdl并且储存结果', loss)
            best_loss = loss
            torch.save(net.state_dict(), 'model/best.mdl')
            savemat('result/RS.mat', {'x': RS.detach().cpu().numpy().argmax(1)})
            savemat('result/RT.mat', {'x': RT.detach().cpu().numpy().argmax(1)+1})

if __name__ == "__main__":

    main()
