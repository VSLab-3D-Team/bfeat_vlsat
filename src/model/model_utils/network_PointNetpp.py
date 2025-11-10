import torch.nn as nn
import torch.nn.functional as F
from src.utils.pointnetpp_utils import PointNetSetAbstraction


class PointNetpp(nn.Module):
    def __init__(self, num_dim, num_channel=9):
        super(PointNetpp, self).__init__()
        in_channel = num_channel
        self.normal_channel = num_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 64], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=64 + 3, mlp=[64, 128, 512], group_all=True)
        self.fc1 = nn.Linear(512, num_dim)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l1_xyz, l1_points)
        x = l3_points.view(B, 512)
        x = self.fc1(x)
        return x, l3_points
