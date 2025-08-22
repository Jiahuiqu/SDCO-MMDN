"""
T1, T2 共享同一个字典的版本，级联在一起求D和V
增加A矩阵在ae网里
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import savemat, loadmat
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AENet(nn.Module):
    def __init__(self, Cin):
        super(AENet,self).__init__()
        self.linear1 = nn.Linear(Cin, 10)
        self.linear2 = nn.Linear(10, 5)
        self.linear3 = nn.Linear(5, 10)
        self.linear4 = nn.Linear(10, Cin)
        self.Activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, V):
        V = self.Activation(self.linear1(V))
        V = self.Activation(self.linear2(V))
        # print('V1.shape:', V.shape)
        V_a = torch.mm(V, V.t())
        V_a = self.softmax(V_a)
        # print('V_a.shape:', V_a.shape)
        V = torch.mm(V_a, V)
        # print('V2.shape:', V.shape)
        V = self.Activation(self.linear3(V))
        V = self.linear4(V)
        return V


class Classifer(nn.Module):
    def __init__(self, DN):
        super(Classifer, self).__init__()

        self.linear1 = nn.Linear(DN, 50)
        self.linear2 = nn.Linear(50, 100)
        self.linear3 = nn.Linear(100, 50)
        self.linear4 = nn.Linear(50, 2)
        self.relu = nn.ReLU()

    def forward(self, V):

        Out = self.relu(self.linear1(V))
        Out = self.relu(self.linear2(Out))
        Out = self.relu(self.linear3(Out))
        Out = self.linear4(Out)

        return Out


class GCNLayer(nn.Module):
    def __init__(self, Cin, CN, N):
        super(GCNLayer, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.N = N
        self.BN = nn.BatchNorm1d(Cin)
        self.Activation = nn.ReLU()
        self.linear1 = nn.Linear(Cin, CN)
        self.DT = nn.Linear(Cin, CN)
        self.DTD = nn.Linear(CN, CN)
        self.AE = AENet(CN)
        self.alpha.data.fill_(0.01)

    def forward(self, H):
        H = self.BN(H)
        V = self.linear1(H)
        V = self.Activation(V)
        S = self.AE(V)
        S = self.Activation(S)
        for i in range(self.N-1):
            V = self.DTD(self.DT(H) + self.alpha * S)
            V = self.Activation(V)
            S = self.AE(V)
            S = self.Activation(S)
        D_W = self.DT.weight.t()
        H_R = F.linear(V, D_W)
        return V, H_R

class GraphOnly(nn.Module):
    def __init__(self, Cin, CN, N):
        super(GraphOnly, self).__init__()
        self.TNet = GCNLayer(Cin, CN, N)
        self.SNet = GCNLayer(Cin, CN, N)
        self.softmax = nn.Softmax(1)

        # self.classiferT = Classifer(CN)
        self.classiferS = Classifer(CN)

    def forward(self, T1, T2, S1, S2):
        """
        :param x_HSI: H*W*C
        :param x_LiDAR: H*W*1
        :return: probability_map
        """

        TF1, R_T1 = self.TNet(T1)
        TF2, R_T2 = self.TNet(T2)
        SF1, R_S1 = self.SNet(S1)
        SF2, R_S2 = self.SNet(S2)
        # GCN层 1 转化为超像素 x_flat 乘以 列归一化Q
        GCN_result_S = SF1 - SF2 # 这里self.norm_row_Q == self.Q

        RS = self.softmax(self.classiferS(GCN_result_S))

        # RS = torch.matmul(self.QS, RS)
        GCN_result_T = TF1 - TF2 # 这里self.norm_row_Q == self.Q
        RT = self.softmax(self.classiferS(GCN_result_T))

        # RT = torch.matmul(self.QT, RT)
        return GCN_result_T, GCN_result_S, RT, RS, R_T1, R_T2, R_S1, R_S2

# 'superpixels_flatten_S2.mat'
# a = loadmat('superpixels_flatten_S1.mat')['x']
# b = loadmat('superpixels_flatten_S2.mat')['x']
# c = loadmat('superpixels_flatten_T1.mat')['x']
# d = loadmat('superpixels_flatten_T2.mat')['x']
#
# a = torch.randn(50,224)
# net = GraphOnly(224,5,4)
# b,c,d,e,f,g,h,i = net(a,a,a,a)