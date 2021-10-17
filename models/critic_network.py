import torch
from torch import nn
import config as c
import torch.nn.functional as F


class KCSLayer(nn.Module):
    # implementation of the Kinematic Chain Space (Wandt et al. https://arxiv.org/abs/1902.09868)
    def __init__(self):
        super(KCSLayer, self).__init__()
        self.C = torch.tensor([
            [1., 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0],
            [-1, 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  1],
            [0,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1],
            [0,  0,  0,  0, -1,  1,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0]]).to(c.device)

        self.number_bones = self.C.shape[1]

    def forward(self, poses3d):
        B = torch.matmul(poses3d.view(-1, 3, 16), self.C)
        Psi = torch.matmul(torch.transpose(B, 1, 2), B)
        return Psi


class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.kcs_layer = KCSLayer()
        n_bones = self.kcs_layer.number_bones

        # the critic network splits in two paths
        # 1) a simple fully connected path
        # 2) the path containing the KCS layer

        # pose path
        self.fc1 = nn.Linear(3*16, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)

        # KCS path
        self.fc5 = nn.Linear(n_bones*n_bones, 1000)
        self.fc6 = nn.Linear(1000, 1000)
        self.fc7 = nn.Linear(1000, 1000)

        self.fc8 = nn.Linear(1100, 100)
        self.fc9 = nn.Linear(100, 1)

    def forward(self, poses3d):
        # pose path
        x1 = self.fc1(poses3d)
        x1 = F.leaky_relu(x1)
        x = self.fc2(x1)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = x + x1
        x = F.leaky_relu(x)
        x = self.fc4(x)

        # KCS path
        psi_vec = self.kcs_layer(poses3d).flatten(start_dim=1)
        psi_vec = self.fc5(psi_vec)
        psi_vec = F.leaky_relu(psi_vec)
        x_psi = self.fc6(psi_vec)
        x_psi = F.leaky_relu(x_psi)
        x_psi = self.fc7(x_psi)
        x_psi = x_psi + psi_vec

        # concatenate both paths and map to output
        x = torch.cat((x, x_psi), dim=1)
        x = self.fc8(x)
        x = F.leaky_relu(x)
        x = self.fc9(x)
        return x
