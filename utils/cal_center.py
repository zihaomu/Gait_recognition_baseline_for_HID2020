import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import distance


def L2_distance(feature1, feature2):
    # the type of feature1 and feature2 is np.array
    if type(feature1) is torch.Tensor:
        feature1 = feature1.detach().numpy()

    if type(feature2) is torch.Tensor:
        feature2 = feature2.detach().numpy()

    return np.linalg.norm(feature1-feature2)


def cuda_dist_tensor(x, y):  # probe_seq_x, gallery_seq_y

    if type(x) is not torch.Tensor:
        x = torch.from_numpy(x).cuda()
    else:
        x = x.cuda()

    if type(y) is not torch.Tensor:
        y = torch.from_numpy(y).cuda().squeeze(1)
    else:
        y = y.cuda().squeeze(1)

    print("size of x : ", x.size())
    print("size of y :",  y.size())
    # y = torch.from_numpy(y).cuda().squeeze(1)
    a = torch.sum(x ** 2, 1).unsqueeze(1)
    b = torch.sum(y ** 2, 1).unsqueeze(1).transpose(0, 1)
    c = 2 * torch.matmul(x, y.transpose(0, 1))
    dist = a + b - c
    dist = torch.sqrt(F.relu(dist))
    return dist.detach().cpu().numpy()


def Vector_module(feature):
    # the type of feature is np.array
    if type(feature) is torch.Tensor:
        feature = feature.detach().numpy()
    return np.linalg.norm(feature)
