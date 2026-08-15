import sys
import importlib.util
import pkgutil
import pytorch3d.ops
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from pytorch3d.ops import knn_points
from resnet import ResNet, InceptionResNet, InceptionResNetMask, ResNetMask, InceptionResNetLight, ResNetBlock
import math
import numpy as np
import open3d
try:
    import torchac
except ImportError:
    torchac = None
import struct

def PSNR(pc1, pc2, n1, return_all=False, peak_value=4095):
    pc1, pc2 = pc1.to(torch.float32), pc2.to(torch.float32)
    dist1, knn1, _ = knn_points(pc1, pc2, K=4)  # neighbors of pc1 from pc2
    dist2, knn2, _ = knn_points(pc2, pc1, K=4)  # neighbors of pc2 from pc1
    mask1 = (dist1 == dist1[:, :, :1])
    mask2 = (dist2 == dist2[:, :, :1])
    dist = max(dist1[:, :, 0].mean(), dist2[:, :, 0].mean())  # dists from knn_points are squared dists
    cd = max(dist1[:, :, 0].sqrt().mean(), dist2[:, :, 0].sqrt().mean())
    # print(pc1, pc2)
    d1_psnr = 10*math.log(3*peak_value*peak_value/dist)/math.log(10)
    d1_psnr_1 = 10 * math.log(3 * peak_value * peak_value / dist1[:, :, 0].mean()) / math.log(10)
    d1_psnr_2 = 10 * math.log(3 * peak_value * peak_value / dist2[:, :, 0].mean()) / math.log(10)
    knn1_ = knn1.reshape(-1)
    n1_src = (n1.unsqueeze(2).repeat(1, 1, 4, 1)*(mask1.unsqueeze(-1))).reshape(-1, 3)
    n2 = torch.zeros_like(pc2.squeeze(0), dtype=torch.float32)
    n2.index_add_(0, knn1_, n1_src)
    n2 = n2.reshape(1, -1, 3)

    n2_counter = torch.zeros(pc2.size()[1], dtype=torch.float32, device=pc2.device)
    counter_knn1 = knn1.reshape(-1)
    n1_counter_src = mask1.reshape(-1).to(torch.float32)
    n2_counter.index_add_(0, counter_knn1, n1_counter_src)
    n2_counter = n2_counter.reshape(1, -1, 1)
    n2_counter += 0.00000001

    n2 /= n2_counter

    v2 = index_points(pc1, knn2) - pc2.unsqueeze(2)
    n2_ = index_points(n1, knn2)
    n21 = (n2_*(mask2.unsqueeze(-1))).sum(dim=2) / (mask2.sum(dim=-1, keepdim=True))
    n2 += (n2_counter < 0.0001) * n21

    d2_ = (((v2*n2_).sum(dim=-1).square()*mask2).sum(dim=-1)/mask2.sum(dim=-1)).mean()
    v1 = index_points(pc2, knn1) - pc1.unsqueeze(2)
    n1_ = index_points(n2, knn1)
    d1_ = (((v1 * n1_).sum(dim=-1).square() * mask1).sum(dim=-1) / mask1.sum(dim=-1)).mean()
    dist_ = max(d1_, d2_)
    d2_psnr = 10*math.log(3*peak_value*peak_value/dist_)/math.log(10)
    # print(d1_psnr, d2_psnr)
    if return_all:
        return d1_psnr_1, d1_psnr_2, d1_psnr, d2_psnr, cd.item()
    else:
        return d1_psnr, d2_psnr, cd.item()

def segment_and_knn(x1, x2, K, bounding_box=None, bias=16):
    ''' 
    fast knn algorithm
    '''
    seg = [[i, j, k] for i in [True, False] for j in [True, False] for k in [True, False]]
    x1_batch = []
    x2_batch = []
    x1_length = []
    x2_length = []
    x1_sorted = []
    idx1_batch = []
    idx1_base = torch.arange(x1.shape[0], dtype=torch.long, device=x1.device)
    idx2_batch = []
    idx2_base = torch.arange(x2.shape[0], dtype=torch.long, device=x2.device)
    x = torch.cat([x1, x2], dim=0)
    x_min = x.min(dim=0, keepdim=True)[0]
    x_max = x.max(dim=0, keepdim=False)[0]
    x1 = x1 - x_min
    x2 = x2 - x_min
    x1_org = x1.clone()
    if bounding_box is None:
        bounding_box = torch.div(x_max, 2, rounding_mode='floor')
    bias = torch.ceil(bias/1024*(x_max-x_min).max())  # offset for splitting boundary
    for s in seg:
        # bias = [0 for b in s]
        b = [x*2*bias-bias for x in s]  # Convert True/False to 1/-1, b is +/-bias
        mask1 = ((x1[:, 0]<bounding_box[0])^s[0])&((x1[:, 1]<bounding_box[1])^s[1])&((x1[:, 2]<bounding_box[2])^s[2])
        if torch.sum(mask1)==0:
            continue
        mask2 = ((x2[:, 0]<bounding_box[0]-b[0])^s[0])&((x2[:, 1]<bounding_box[1]-b[1])^s[1])&((x2[:, 2]<bounding_box[2]-b[2])^s[2])
        counter = 1
        while torch.sum(mask2)==0:
            bias1 = bias+counter*bias
            b1 = [x*2*bias1-bias1 for x in s]
            mask2 = ((x2[:, 0]<bounding_box[0]-b1[0])^s[0])&((x2[:, 1]<bounding_box[1]-b1[1])^s[1])&((x2[:, 2]<bounding_box[2]-b1[2])^s[2])
            counter += 1
        x1_s = x1[mask1]
        if x1_s.shape[0] == 0:
            continue
        x2_s = x2[mask2]
        idx1 = idx1_base[mask1]
        idx1_batch.append(idx1)
        idx2 = idx2_base[mask2]
        idx2_batch.append(idx2)
        x1_length.append(x1_s.shape[0])
        x2_length.append(x2_s.shape[0])
        x1_batch.append(x1_s)
        x2_batch.append(x2_s)
        x1_sorted.append(x1_s)
    x1 = torch.cat(x1_sorted, dim=0)  # (sum(N_i), 3)
    idx1 = torch.cat(idx1_batch, dim=0)  # (sum(N_i),)

    x1_length = torch.tensor(x1_length, dtype=torch.long).to(x1.device)
    x2_length = torch.tensor(x2_length, dtype=torch.long).to(x2.device)
    max_x1_len = max(x1_length)
    max_x2_len = max(x2_length)
    padded_x1 = torch.stack([F.pad(x, (0, 0, 0, max_x1_len - x.shape[0])) for x in x1_batch])  # (B, max_x1, 3)
    padded_x2 = torch.stack([F.pad(x, (0, 0, 0, max_x2_len - x.shape[0])) for x in x2_batch])  # (B, max_x2, 3)
    padded_idx2 = torch.stack([F.pad(idx, (0, max_x2_len - idx.shape[0])) for idx in idx2_batch])  # (B, max_x2)

    d_, idx_, _ = knn_points(padded_x1, padded_x2, x1_length, x2_length, K=K, return_sorted=True)

    # print(padded_x1.size(), padded_x2.size(), x1.size(), x2.size())

    # Unpack and mask
    final_dists = []
    final_idx = []
    for i in range(len(x1_length)):
        N = x1_length[i]
        N2 = x2_length[i]
        d = d_[i, :N]
        idx = idx_[i, :N]     # (N, K)
        
        # Convert local idx to global by offsetting with base of x2_segs
        assert idx.max() <= N2, "Index exceeds the number of points in x2"
        idx = padded_idx2[i, :N2][idx]
        assert (idx2_batch[i] == padded_idx2[i, :N2]).all()
        final_dists.append(d)
        final_idx.append(idx)


    final_dists = torch.cat(final_dists, dim=0)  # (sum(N_i), K)
    final_idx = torch.cat(final_idx, dim=0)      # (sum(N_i), K)

    x1, x1_org = x1.to(torch.float64), x1_org.to(torch.float64)
    x1_idx_inverse = torch.argsort(idx1)
    x1, final_dists, final_idx = x1[x1_idx_inverse], final_dists[x1_idx_inverse], final_idx[x1_idx_inverse]
    
    assert (x1==x1_org).all(), "x1 and x1_org should be the same after sorting"

    return final_dists, final_idx, x1

def sort_by_coor_sum(f, stride=None):
    if stride is None:
        stride = f.tensor_stride[0]
    xyz = f.C.clone()
    maximum = xyz.max() + 1
    xyz, maximum = xyz.long(), maximum.long()
    # non-negative (positive values only for sort)
    if xyz.min() < 0:
        min_value = xyz.min()
        xyz[:, 1:] = xyz[:, 1:] - min_value
        maximum = maximum - min_value
    assert xyz.min() >= 0 and xyz.max() - xyz.min() < maximum
    # coord to 1D
    coor_sum = xyz[:, 0] * maximum * maximum * maximum \
               + xyz[:, 1] * maximum * maximum \
               + xyz[:, 2] * maximum \
               + xyz[:, 3]
    # sort
    _, idx = coor_sum.sort()
    xyz_, feature_ = f.C[idx], f.F[idx]
    f_ = ME.SparseTensor(feature_, coordinates=xyz_, tensor_stride=stride, device=f.device)

    assert (sort_sparse_tensor(f).C == f_.C).all()  ### todo: comparison
    assert (sort_sparse_tensor(f).F == f_.F).all()
    return f_

def sample_offset_on_normal_plane(normal, radius=0.5, eps=1e-8):
    device = normal.device
    dtype = normal.dtype
    N = normal.shape[0]

    normal_norm = torch.norm(normal, dim=-1, keepdim=True)
    valid = normal_norm.squeeze(-1) > eps

    offset = torch.zeros((N, 3), device=device, dtype=dtype)

    if valid.sum() == 0:
        return offset

    n = torch.zeros_like(normal)
    n[valid] = normal[valid] / normal_norm[valid].clamp_min(eps)

    # 构造 tangent basis
    ref = torch.zeros_like(n)
    ref[:, 2] = 1.0

    bad = n[:, 2].abs() > 0.9
    ref[bad] = torch.tensor(
        [0.0, 1.0, 0.0],
        device=device,
        dtype=dtype,
    )

    t1 = torch.cross(n, ref, dim=-1)
    t1 = F.normalize(t1, dim=-1, eps=eps)

    t2 = torch.cross(n, t1, dim=-1)
    t2 = F.normalize(t2, dim=-1, eps=eps)

    rho = torch.sqrt(torch.rand((N, 1), device=device, dtype=dtype))
    theta = 2.0 * math.pi * torch.rand((N, 1), device=device, dtype=dtype)

    u = rho * torch.cos(theta)
    v = rho * torch.sin(theta)

    offset_valid = radius * (u * t1 + v * t2)

    def scale_offset_into_box(offset, half=0.5, eps=1e-8):
        max_abs = offset.abs().max(dim=-1, keepdim=True).values  # [N, 1]
        scale = torch.clamp(half / max_abs.clamp_min(eps), max=1.0)
        return offset * scale
    offset_valid = scale_offset_into_box(offset_valid, half=0.5)
    offset[valid] = offset_valid[valid]

    return offset

def set_stride(x, stride):
    return ME.SparseTensor(x.F, coordinates=x.C, tensor_stride=stride)

def sparse_tensor_upsample(x, y):
    x = sort_sparse_tensor(x)
    y_sorted = sort_sparse_tensor(y)
    target_stride = y.tensor_stride[0]
    pooling = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3, expand_coordinates=True) if x.tensor_stride[0]>target_stride else ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
    while x.tensor_stride[0]!=target_stride:
        if x.tensor_stride[0]//2==target_stride or x.tensor_stride[0]*2==target_stride:
            x = pooling(x, y_sorted.C)
            break
        else:
            x = pooling(x)
    if x.size()[0] != y.size()[0]:
        yMask = ME.SparseTensor(torch.ones_like(y.F[:,0:1], dtype=torch.float32), coordinates=y.C, device=y.device)
        mask = yMask.features_at_coordinates(x.C.to(torch.float32)).squeeze()>0.9
        x = ME.SparseTensor(x.F[mask], coordinates=x.C[mask], tensor_stride=x.tensor_stride[0], device=x.device)
    x = sort_sparse_tensor(x, y)
    return x

def coordinate_sort_by_coor_sum(xyz_in):
    maximum = xyz_in.max() + 1
    xyz, maximum = xyz_in.clone().long(), maximum.long()
    # non-negative (positive values only for sort)
    if xyz.min() < 0:
        min_value = xyz.min()
        xyz[:, 1:] = xyz[:, 1:] - min_value
        maximum = maximum - min_value
    assert xyz.min() >= 0 and xyz.max() - xyz.min() < maximum
    # coord to 1D
    coor_sum = xyz[:, 0] * maximum * maximum * maximum \
               + xyz[:, 1] * maximum * maximum \
               + xyz[:, 2] * maximum \
               + xyz[:, 3]
    _, idx = coor_sum.sort()
    xyz_ = xyz_in[idx].to(torch.float32)
    return xyz_


def index_points(points, idx):
    """
    Input:
        points: input points data, [B*C, N1, 1]
        idx: sample index data, [B*C, N2, K]
    Return:
        new_points:, indexed points data, [B*C, N2, K, 1]
    """
    device = points.device
    B = points.shape[0]  # B*C
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # B*C,1,1
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # 1,N2,K
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # B*C,1,1 -> B*C,N2,K
    new_points = points[batch_indices, idx, :]
    return new_points


def quant(x, training=False, qs=1, ste=False):
    if training and not ste:
        compressed_x = x + torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5) * qs
    elif training and ste:
        compressed_x = (torch.round(x / qs)*qs-x).detach()+x
    else:
        compressed_x = torch.round(x / qs) * qs
    return compressed_x

def ravel_multi_index(coords, step):
    coords = coords.long()
    step = step.long()
    coords_sum = coords[:, 3] \
                    + coords[:, 2] * step \
                    + coords[:, 1] * step * step \
                    + coords[:, 0] * step * step * step
    return coords_sum

def get_target_by_sp_tensor(out, target_sp_tensor):
    with torch.no_grad():
        def ravel_multi_index(coords, step):
            coords = coords.long()
            step = step.long()
            coords_sum = coords[:, 3] \
                         + coords[:, 2] * step \
                         + coords[:, 1] * step * step \
                         + coords[:, 0] * step * step * step
            return coords_sum

        # non-negative
        out_coords, target_coords = out.C.clone(), target_sp_tensor.C.clone()
        min_value = torch.min(out_coords.min(), target_coords.min())  ### todo: must the same min_value!!!
        if min_value < 0:
            out_coords[:, 1:] -= min_value
            target_coords[:, 1:] -= min_value

        step = max(out_coords.max(), target_coords.max()) + 1

        out_sp_tensor_coords_1d = ravel_multi_index(out_coords, step)
        in_sp_tensor_coords_1d = ravel_multi_index(target_coords, step)

        # test whether each element of a 1-D array is also present in a second array.
        target = torch.isin(out_sp_tensor_coords_1d, in_sp_tensor_coords_1d)

        assert (isin(out.C, target_sp_tensor.C) == target).all()  ### todo: comparison
    return target


def write_point_cloud(filename, c, f, stride=1, yuv=1):
    motion_pcd = open3d.geometry.PointCloud()
    motion_pcd.points = open3d.utility.Vector3dVector((c//stride).detach().cpu().numpy())
    if f is not None:
        color = f.detach().cpu().numpy()
        from dataset_lossy_v1 import yuv2rgb
        color = yuv2rgb(color) if yuv else color
        motion_pcd.colors = open3d.utility.Vector3dVector(color)
    open3d.io.write_point_cloud(filename, motion_pcd, write_ascii=True)
    with open(filename, 'r') as f:
        lines = f.readlines()
    with open(filename, 'w') as f:
        for line in lines:
            line = line.replace('double', 'float')
            f.write(line)

def write_ply_open3d_normal(filename, coords, dtype='int32', knn=20):
    pcd = open3d.geometry.PointCloud() 
    pcd.points = open3d.utility.Vector3dVector(coords.astype(dtype)) 
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamKNN(knn=knn)) 
    open3d.io.write_point_cloud(filename, pcd, write_ascii=True)
    f = open(filename)
    lines = f.readlines() 
    lines[4] = 'property float x\n' 
    lines[5] = 'property float y\n' 
    lines[6] = 'property float z\n' 
    lines[7] = 'property float nx\n' 
    lines[8] = 'property float ny\n' 
    lines[9] = 'property float nz\n' 
    fo = open(filename, "w")
    fo.writelines(lines) 
    return

def read_point_cloud(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    C = []
    F = []
    for line in lines:
        if '0'<=line[0]<='9':
            s = line.split(' ')
            C.append([float(n) for n in s[:3]])
            F.append([float(n) for n in s[3:]])
    C = np.asarray(C)
    F = np.asarray(F)
    return C, F

def make_layer(block, block_layers, channels, kernel_size=3):
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels, kernel_size=kernel_size))
    return nn.Sequential(*layers)

def rewrite_double(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    with open(filename, 'w') as f:
        for line in lines:
            line = line.replace('double', 'float')
            f.write(line)

def keep_adaptive(out, coords_nums, rho=1.0):
    ''' keep_adaptive_kplus1 '''
    with torch.no_grad():
        keep = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        #  get row indices per batch.
        # row_indices_per_batch = out.coords_man.get_row_indices_per_batch(out.coordinate_map_key)
        row_indices_per_batch = out._batchwise_row_indices

        for row_indices, ori_coords_num in zip(row_indices_per_batch, coords_nums):
            coords_num = min(len(row_indices), ori_coords_num * rho)  # select top k points.
            out_cls_F = torch.sigmoid(out.F[row_indices].squeeze())  # out probability of current batch
            # top-k local
            _, indices = torch.topk(out_cls_F.reshape(-1,8), k=1)
            indices += (torch.arange(0, len(indices))*8).reshape(-1,1).to(indices.device)
            indices = indices.reshape(-1)
            out_cls_F[indices] = 1
            # top-k global
            values, indices = torch.topk(out_cls_F, int(coords_num))
            # keep for current batch
            keep[row_indices[indices]] = True
    return keep

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Initialize weights uniformly from -0.01 to 0.01
        nn.init.uniform_(m.weight, a=-0.0005, b=0.0005)  
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Initialize bias to zero

def init_weights_ME(m):
    if isinstance(m, ME.MinkowskiConvolution):
        # Initialize weights uniformly from -0.01 to 0.01
        nn.init.uniform_(m.kernel, a=-0.0005, b=0.0005)  
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Initialize bias to zero

def prune(f1, f2):
    if type(f2)!=ME.SparseTensor:
        f2 = ME.SparseTensor(torch.ones_like(f2[:, :1]), coordinates=f2, tensor_stride=f1.tensor_stride[0])
    mask = get_target_by_sp_tensor(f1, f2)
    pruning = ME.MinkowskiPruning()
    out = pruning(f1, mask.to(f1.device))
    return out

def array2vector(array, step=None):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long().clone(), step.long().clone()
    if array.min() < 0:
        min_value = array.min()
        array = array - min_value
        step = step - min_value

    assert array.min() >= 0 and array.max() - array.min() < step
    array, step = array.long(), step.long()
    vector = sum([array[:, i] * (step ** (array.shape[-1] - 1 - i)) for i in
                  range(array.shape[-1])])  ### todo: can be step**(array.shape[-1]-i)

    return vector


def isin(data, ground_truth):
    """ Input data and ground_truth are torch tensor of shape [N, D].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is in `ground_truth` and False otherwise.
    """
    data = data.clone()
    ground_truth = ground_truth.clone()
    device = data.device
    if len(ground_truth) == 0:
        return torch.zeros([len(data)]).bool().to(device)
    # positive value
    min_value = torch.min(data.min(), ground_truth.min())
    if min_value < 0:
        data[:, 1:] -= min_value
        ground_truth[:, 1:] -= min_value
    #
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = torch.isin(data.to(device), ground_truth.to(device))

    return mask


def istopk(data, nums, rho=1.0):
    """ Input data is sparse tensor and nums is a list of shape [batch_size].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is the top k (=nums*rho) value and False otherwise.
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N * rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)  # must CPU.
        mask[row_indices[indices]] = True
    return mask


def create_new_sparse_tensor(coordinates, features, tensor_stride, dimension, device):
    sparse_tensor = ME.SparseTensor(features=features,
                                    coordinates=coordinates,
                                    tensor_stride=tensor_stride,
                                    device=device)

    return sparse_tensor


def sort_sparse_tensor(sparse_tensor, target=None):
    """ Sort points in sparse tensor according to their coordinates or the coords of target
    """
    if target is not None and (sparse_tensor.C == target.C).all():
        return ME.SparseTensor(features=sparse_tensor.F,
                               coordinate_map_key=target.coordinate_map_key,
                               coordinate_manager=target.coordinate_manager,
                               device=target.device)

    # positive value
    coords = sparse_tensor.C.clone()
    min_value = coords.min()
    if min_value < 0: coords[:, 1:] -= min_value
    # sort
    indices = torch.argsort(array2vector(coords, coords.max() + 1))
    out_coords = sparse_tensor.C[indices]
    out_feats = sparse_tensor.F[indices]
    out = create_new_sparse_tensor(coordinates=out_coords, features=out_feats,
                                   tensor_stride=sparse_tensor.tensor_stride,
                                   dimension=sparse_tensor.D, device=sparse_tensor.device)
    if target is not None:
        # positive value
        target_coords = target.C.clone()
        min_value = target_coords.min()
        if min_value < 0: target_coords[:, 1:] -= min_value
        # sort
        target_indices = torch.argsort(array2vector(target_coords, target_coords.max() + 1))
        inverse_indices = target_indices.sort()[1]
        assert (out_coords[inverse_indices] == target.C).all()
        out = ME.SparseTensor(features=out_feats[inverse_indices],
                              coordinate_map_key=target.coordinate_map_key,
                              coordinate_manager=target.coordinate_manager,
                              device=target.device)

    return out


def topkplus1_coordinates(scores, coordinates, bounding_box_size, target):
    """Select the highest-scoring coordinate in each parent block."""
    block_size = int(bounding_box_size)
    spatial_coordinates = coordinates[:, 1:]
    block_origins = torch.div(
        spatial_coordinates, block_size, rounding_mode='floor'
    ) * block_size
    block_coordinates = torch.cat([coordinates[:, :1], block_origins], dim=1)
    unique_blocks, block_indices = torch.unique(
        block_coordinates, dim=0, sorted=True, return_inverse=True
    )

    scores = scores.reshape(-1)
    best_scores = torch.full(
        (unique_blocks.size(0),), -torch.inf,
        dtype=scores.dtype, device=scores.device
    )
    best_scores.scatter_reduce_(
        0, block_indices, scores, reduce='amax', include_self=True
    )

    local_coordinates = torch.remainder(spatial_coordinates, block_size).long()
    local_indices = (
        local_coordinates[:, 0] * block_size ** 2
        + local_coordinates[:, 1] * block_size
        + local_coordinates[:, 2]
    )
    candidate_indices = torch.where(
        scores == best_scores[block_indices],
        local_indices,
        torch.full_like(local_indices, -1),
    )
    best_local_indices = torch.full(
        (unique_blocks.size(0),), -1,
        dtype=torch.long, device=scores.device
    )
    best_local_indices.scatter_reduce_(
        0, block_indices, candidate_indices, reduce='amax', include_self=True
    )

    local_coordinates = torch.stack([
        best_local_indices // block_size ** 2,
        (best_local_indices % block_size ** 2) // block_size,
        best_local_indices % block_size,
    ], dim=1).to(torch.int32)
    selected_coordinates = local_coordinates + unique_blocks[:, 1:].to(torch.int32)
    selected = ME.SparseTensor(selected_coordinates, coordinates=unique_blocks)
    return sort_sparse_tensor(selected, target).F.to(torch.int32)

def upsample_to_top(x, stride=None, coordinates=None):
    # x: input sparse tensor, will upsample to stride=1.
    upsample = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3, expand_coordinates=True)
    if stride is None:
        stride = x.tensor_stride[0]
    while stride!=1:
        if coordinates is None:
            x = upsample(x)
        elif stride==2:
            x = upsample(x, coordinates)
        stride //= 2
    return x

def quat2rotmat(q):
    # q: shape (N,4) – quaternion in (w, x, y, z) format
    w, x, y, z = q.unbind(dim=-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = torch.stack((
        ww+xx-yy-zz,  2*(xy-wz),     2*(xz+wy),
        2*(xy+wz),    ww-xx+yy-zz,   2*(yz-wx),
        2*(xz-wy),    2*(yz+wx),     ww-xx-yy+zz
    ), dim=-1).reshape(-1,9)
    return R           # shape (N,3,3)

def gaussian_pdf_aniso(xyz: torch.Tensor,
                    s:   torch.Tensor,
                    rotmat: torch.Tensor,
                    beta: torch.Tensor=None) -> torch.Tensor:
    """
    xyz : (N,3)   – offsets in world coords
    s   : (N,3)   – local-axis scales > 0
    rotmat: (N,3,3) – rotation matrices
    returns p : (N,) – PDF values
    """
    N = xyz.shape[0]
    R = rotmat
    x_local = torch.bmm(R.transpose(1,2), xyz.unsqueeze(-1)).squeeze(-1)  # (N,3)

    inv_s2   = 1.0 / (s*s)               # (N,3)
    mahalan2 = (x_local**2 * inv_s2).sum(dim=-1)   # (N,)
    if beta is not None:
        beta = beta.squeeze()
        mahalan2 = mahalan2 ** ((2+beta) / 2.0) 
    detS     = s.prod(dim=-1)                           # (N,)

    norm_const = ( (2*math.pi)**1.5 ) * detS            # (N,)
    p = torch.exp(-0.5 * mahalan2)

    return p

class Bitparm(nn.Module):
    # save params
    def __init__(self, channel, dimension=4, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        para = [1 for i in range(dimension)]
        para[dimension - 1] = -1
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(para), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(para), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(para), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)


class BitEstimator(nn.Module):
    '''
    factorized
    '''
    def __init__(self, channel, dimension=3):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel, dimension=dimension)
        self.f2 = Bitparm(channel, dimension=dimension)
        self.f3 = Bitparm(channel, dimension=dimension)
        self.f4 = Bitparm(channel, dimension=dimension, final=True)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)


class HyperPriorMultiStage(nn.Module):
    def __init__(self, input, hidden, output, resnet=InceptionResNetLight, block_layers=2, ctxSize=0, normalizeConfig=0, hyper=True, adaptive_quant=False, dimBeforeDownsample=0, downsample=False, stage=1):
        super(HyperPriorMultiStage, self).__init__()
        self.input, self.hidden, self.output = input, hidden, output
        if not hyper:
            self.factorizedModel = BitEstimator(input)
        self.hyper = hyper
        self.ctxSize = ctxSize
        if ctxSize==0:
            # if no context (e.g., for the coarsest layer), use a learnable variable as context. Note that this needs no compression since it's fixed for all.
            self.ctx = nn.Parameter(torch.zeros(1, 16, dtype=torch.float32), requires_grad=True)
            ctxSize = 16
        self.ctxMerge = nn.Sequential(
            make_layer(resnet, block_layers, ctxSize),
            ME.MinkowskiConvolution(in_channels=ctxSize, out_channels=input+input, kernel_size=3, stride=1, bias=True, dimension=3),
            )
        self.adaptive_quant = adaptive_quant
        if self.adaptive_quant:
            self.qpGenerator = nn.Sequential(
            make_layer(resnet, block_layers, ctxSize),
                ME.MinkowskiConvolution(in_channels=ctxSize, out_channels=input+input, kernel_size=3, stride=1, bias=True, dimension=3),
            )

        self.stage = stage
        if stage>1:
            self.stageCtxExtract = ME.MinkowskiConvolution(in_channels=input, out_channels=hidden, kernel_size=5, stride=1, bias=True, dimension=3)
            self.ctxMergeFinal = nn.Sequential(
                ME.MinkowskiConvolution(in_channels=input+input+hidden, out_channels=input+input+hidden, kernel_size=1, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(),
                ME.MinkowskiConvolution(in_channels=input+input+hidden, out_channels=input+input+hidden, kernel_size=1, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(),
                ME.MinkowskiConvolution(in_channels=input+input+hidden, out_channels=input+input, kernel_size=1, stride=1, bias=True, dimension=3),
            )


    def get_stage(self, x, stage):
        x_c = x.C[:, 1:]
        x_c = x_c//x.tensor_stride[0]
        x_c = x_c%2
        x_c_idx = x_c[:, 0]+x_c[:, 1]*2+x_c[:, 2]*4
        if self.stage==2:
            stage_list = [[0, 3, 5, 6], [1, 2, 4, 7]]
        elif self.stage==4:
            stage_list = [[0, 7], [3, 4], [1, 6], [2, 5]]
        elif self.stage==8:
            stage_list = [[0], [3], [5], [6], [1], [2], [4], [7]]
        stage_idx = stage_list[stage]
        stage_x_c, stage_x_f = [], []
        for i in stage_idx:
            stage_x_c.append(x.C[x_c_idx==i])
            stage_x_f.append(x.F[x_c_idx==i])
        stage_x_c = torch.cat(stage_x_c, dim=0)
        stage_x_f = torch.cat(stage_x_f, dim=0)
        stage_x = ME.SparseTensor(stage_x_f, coordinates=stage_x_c, coordinate_manager=x.coordinate_manager)
        return stage_x
            
    
    def forward(self, x, training=True, gaussion=True, require_quant=True, ctx=None, bar=False):
        ctxCat = ME.SparseTensor(torch.zeros_like(x.F[:, :0]), coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        device = x.device
        N = x.F.size(0)
        if ctx is not None:
            for i, c in enumerate(ctx):
                if c is None:
                    continue
                c_up = sparse_tensor_upsample(c, x)
                if c.F.size(1) == 32:
                    torch.save({
                        'source_coordinates': c.C.cpu(),
                        'source_features': c.F.cpu(),
                        'upsampled_coordinates': c_up.C.cpu(),
                        'upsampled_features': c_up.F.cpu(),
                    }, f'v1_context_{c.tensor_stride[0]}_to_{x.tensor_stride[0]}.pt')
                ctxCat = ME.cat(ctxCat, c_up)
        else:
            ctxCat = ME.SparseTensor(self.ctx.expand(x.F.size(0), -1), coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        mu_sigma = self.ctxMerge(ctxCat)

        assert (mu_sigma.C==x.C).all()

        if require_quant and not self.adaptive_quant:
            x_hat_F = quant(x.F, training=training, qs=1)
            x_ada_F = x.F
            x_bar_F = quant(x.F, training=training, qs=1, ste=True)
        elif require_quant and self.adaptive_quant:
            qEnc_qDec = self.qpGenerator(ctxCat)
            qEnc_F, qDec_F = torch.split(qEnc_qDec.F, [qEnc_qDec.size()[1]//2, qEnc_qDec.size()[1]//2], dim=1)
            x_ada_F = x.F*(1+qEnc_F)
            x_bar_F = quant(x_ada_F, training=training, qs=1, ste=True)
            x_hat_F = quant(x_ada_F, training=training, qs=1, ste=False)
        else:
            x_hat_F = x.F
            x_ada_F = x.F
            x_bar_F = x.F

        x_hat = ME.SparseTensor(x_hat_F, coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        x_bar = ME.SparseTensor(x_bar_F, coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        x_ada = ME.SparseTensor(x_ada_F, coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)

        if self.stage>1:
            # finetune mu_sigma based on multi-stage context.
            mu_sigma_stage = ME.SparseTensor(torch.zeros_like(mu_sigma.F), coordinate_manager=mu_sigma.coordinate_manager, coordinate_map_key=mu_sigma.coordinate_map_key)
            mu_sigma_input = ME.SparseTensor(torch.cat([mu_sigma.F, torch.zeros(N, self.hidden, dtype=torch.float32, device=device)], dim=1), coordinate_manager=mu_sigma.coordinate_manager, coordinate_map_key=mu_sigma.coordinate_map_key)
            x_bar_rec = ME.SparseTensor(torch.zeros_like(x.F), coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
            x_hat_rec = ME.SparseTensor(torch.zeros_like(x.F), coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)

            for i in range(self.stage):
                coordinates = self.get_stage(x, i).C
                if coordinates.size(0)==0:
                    continue
                ctx = x_bar_rec if bar else x_hat_rec
                stage_ctx = self.stageCtxExtract(ctx, coordinates)
                stage_ctx = ME.SparseTensor(torch.cat([torch.zeros(stage_ctx.size()[0], mu_sigma.size()[1], dtype=torch.float32, device=device), stage_ctx.F], dim=1), coordinates=coordinates, coordinate_manager=mu_sigma_input.coordinate_manager)
                mu_sigma_input = mu_sigma_input + stage_ctx
                mu_sigma_input = sort_sparse_tensor(mu_sigma_input, mu_sigma)


                stage_x_bar = x_bar.features_at_coordinates(coordinates.to(torch.float32))
                stage_x_bar = ME.SparseTensor(stage_x_bar, coordinates=coordinates, coordinate_manager=x_bar_rec.coordinate_manager)
                x_bar_rec = x_bar_rec + stage_x_bar
                assert (x_bar_rec.C==x.C).all()
                x_bar_rec = ME.SparseTensor(x_bar_rec.F, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)

                stage_x_hat = x_hat.features_at_coordinates(coordinates.to(torch.float32))
                stage_x_hat = ME.SparseTensor(stage_x_hat, coordinates=coordinates, coordinate_manager=x_hat_rec.coordinate_manager)
                x_hat_rec = x_hat_rec + stage_x_hat
                x_hat_rec = ME.SparseTensor(x_hat_rec.F, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)

            mu_sigma = self.ctxMergeFinal(mu_sigma_input)+ mu_sigma

        mu_f, sigma_f = torch.split(mu_sigma.F, [mu_sigma.size()[1]//2, mu_sigma.size()[1]//2], dim=1)
        sigma_f = torch.exp(torch.clamp(sigma_f, max=10))


        if not self.hyper:
            x_residual = x_hat_F-mu_f
            p_x = self.factorizedModel(x_residual+0.5)-self.factorizedModel(x_residual-0.5)
            b_x = torch.sum(torch.clamp(-1.0 * torch.log(p_x + 1e-10) / math.log(2.0), 0, 50))
            b_x_nonreduction = torch.sum(torch.clamp(-1.0 * torch.log(p_x + 1e-10) / math.log(2.0), 0, 50), dim=-1, keepdim=True).squeeze(0)
            return {
                'x_hat': x_hat,
                'x_bar': x_bar,
                'p_x': p_x,
                'b_x': b_x,
                'b_x_nonreduction': b_x_nonreduction,
                'x_ada': x_ada,
            }
        if gaussion:
            distribution = torch.distributions.normal.Normal(mu_f, sigma_f)
        else:
            distribution = torch.distributions.laplace.Laplace(mu_f, sigma_f)
        # p_x = distribution.cdf(x_hat_beforeQuant_F+0.5)-distribution.cdf(x_hat_beforeQuant_F-0.5)
        p_x = distribution.cdf(x_hat_F+0.5)-distribution.cdf(x_hat_F-0.5)
        b_x = torch.sum(torch.clamp(-1.0 * torch.log(p_x + 1e-10) / math.log(2.0), 0, 50))
        b_x_nonreduction = torch.sum(torch.clamp(-1.0 * torch.log(p_x + 1e-10) / math.log(2.0), 0, 50), dim=-1, keepdim=True).squeeze(0)
        b = b_x
        if self.adaptive_quant:
            x_hat_F = x_hat_F*(1+qDec_F)
            x_bar_F = x_bar_F*(1+qDec_F)
        x_hat = ME.SparseTensor(x_hat_F, coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        x_bar = ME.SparseTensor(x_bar_F, coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        x_ada = ME.SparseTensor(x_ada_F, coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)

        return {
            'x_hat': x_hat,
            'x_ada': x_ada,
            'x_bar': x_bar,
            'p_x': p_x,
            'b_x': b_x,
            'b_x_nonreduction': b_x_nonreduction,
        }
    
    @torch.no_grad()
    def _build_hyper_context(self, x, ctx=None):
        ctxCat = ME.SparseTensor(torch.zeros_like(x.F[:, :0]),
                                 coordinate_manager=x.coordinate_manager,
                                 coordinate_map_key=x.coordinate_map_key)
        if ctx is not None:
            for c in ctx:
                if c is None:
                    continue
                ctxCat = ME.cat(ctxCat, sparse_tensor_upsample(c, x))
        else:
            ctxCat = ME.SparseTensor(self.ctx.expand(x.F.size(0), -1),
                                     coordinate_manager=x.coordinate_manager,
                                     coordinate_map_key=x.coordinate_map_key)
        mu_sigma = self.ctxMerge(ctxCat)
        assert (mu_sigma.C == x.C).all()
        return ctxCat, mu_sigma

    def _stage_coordinates(self, x, stage_id):
        return self.get_stage(x, stage_id).C

    def _features_at(self, sp_tensor, coordinates):
        return sp_tensor.features_at_coordinates(coordinates.to(torch.float32))

    def _make_torchac_cdf(self, mu_f, sigma_f, symbol_min, symbol_max, gaussion=True):
        """Build finite-support float CDF for torchac.

        torchac requires CDF shape (..., Lp), symbols shape (...), CPU float32
        CDF, and CPU int16 symbols. We map real latent values
        [symbol_min, symbol_max] to torchac indices [0, L-1].
        """
        if torchac is None:
            raise ImportError('torchac is not installed. Install it with: pip install torchac')

        symbol_min = int(symbol_min)
        symbol_max = int(symbol_max)
        L = symbol_max - symbol_min + 1
        if L <= 0:
            raise ValueError('symbol_max must be >= symbol_min')
        # torchac symbols are int16 on the Python side, so the largest index
        # must fit in signed int16.
        if L > 32768:
            raise ValueError(
                f'torchac requires int16 symbols, but symbol range has {L} values. '
                'Use a narrower symbol range or split the latent into several streams.'
            )

        mu_f = mu_f.to(torch.float32)
        sigma_f = sigma_f.to(torch.float32).clamp_min(1e-9)
        edges = torch.arange(symbol_min, symbol_max + 2,
                             device=mu_f.device, dtype=torch.float32) - 0.5

        if not self.hyper:
            raise NotImplementedError('Factorized model only used for training, not for torchac encoding/decoding.')
        else:
            if gaussion:
                distribution = torch.distributions.normal.Normal(mu_f.unsqueeze(-1),
                                                                 sigma_f.unsqueeze(-1))
            else:
                distribution = torch.distributions.laplace.Laplace(mu_f.unsqueeze(-1),
                                                                   sigma_f.unsqueeze(-1))
            raw_cdf = distribution.cdf(edges.view(1, 1, -1))

        # Convert the infinite-support model to the finite alphabet required by
        # torchac. This preserves the relative bin probabilities inside the
        # transmitted support and forces cdf[...,0]=0, cdf[...,-1]=1.
        lo = raw_cdf[..., :1]
        hi = raw_cdf[..., -1:]
        denom = (hi - lo).clamp_min(1e-12)
        cdf = (raw_cdf - lo) / denom
        cdf = torch.nan_to_num(cdf, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        cdf[..., 0] = 0.0
        cdf[..., -1] = 1.0
        return cdf.detach().cpu().contiguous().to(torch.float32)

    def _torchac_encode(self, symbols_F, mu_f, sigma_f, symbol_min, symbol_max, gaussion=True, DBG=False):
        if symbols_F.numel() == 0:
            return b''
        sym = (symbols_F.detach().to(torch.long) - int(symbol_min)).cpu().contiguous()
        if sym.min() < 0 or sym.max() > 32767:
            raise ValueError('torchac symbol index outside int16 range after symbol_min shift.')
        sym = sym.to(torch.int16)
        cdf = self._make_torchac_cdf(mu_f, sigma_f, symbol_min, symbol_max, gaussion=gaussion)
        return torchac.encode_float_cdf(cdf, sym, needs_normalization=True, check_input_bounds=DBG)

    def _torchac_decode(self, bitstream, mu_f, sigma_f, symbol_min, symbol_max, gaussion=True, device=None):
        if mu_f.numel() == 0:
            return torch.empty_like(mu_f, dtype=torch.float32, device=device if device is not None else mu_f.device)
        cdf = self._make_torchac_cdf(mu_f, sigma_f, symbol_min, symbol_max, gaussion=gaussion)
        sym = torchac.decode_float_cdf(cdf, bitstream, needs_normalization=True)
        sym = sym.to(torch.long) + int(symbol_min)
        if device is None:
            device = mu_f.device
        return sym.to(device=device, dtype=torch.float32)

    def _pack_hyperprior_bitstream(self, stream_list, symbol_min, symbol_max, num_points, num_channels):
        # Header:
        # magic(4) version(u16) nstreams(u16) symbol_min(i64) symbol_max(i64)
        # num_points(u32) num_channels(u32) stream_lengths[nstreams](u32) streams...
        header = struct.pack('<4sHHqqII', b'HPTA', 1, len(stream_list),
                             int(symbol_min), int(symbol_max), int(num_points), int(num_channels))
        lengths = b''.join(struct.pack('<I', len(s)) for s in stream_list)
        return header + lengths + b''.join(stream_list)

    def _unpack_hyperprior_bitstream(self, bitstream):
        header_size = struct.calcsize('<4sHHqqII')
        magic, version, nstreams, symbol_min, symbol_max, num_points, num_channels = \
            struct.unpack('<4sHHqqII', bitstream[:header_size])
        if magic != b'HPTA':
            raise ValueError('Invalid HyperPriorMultiStage torchac bitstream magic')
        if version != 1:
            raise ValueError(f'Unsupported HyperPriorMultiStage torchac bitstream version: {version}')
        pos = header_size
        lengths = []
        for _ in range(nstreams):
            lengths.append(struct.unpack('<I', bitstream[pos:pos + 4])[0])
            pos += 4
        streams = []
        for length in lengths:
            streams.append(bitstream[pos:pos + length])
            pos += length
        return {
            'symbol_min': int(symbol_min),
            'symbol_max': int(symbol_max),
            'num_points': int(num_points),
            'num_channels': int(num_channels),
            'streams': streams,
        }

    @torch.no_grad()
    def encode(self, x, training=False, gaussion=True, require_quant=True, ctx=None, bar=False,
               adaptiveQuant=False, prob_up=None, prob_onehot_up=None,
               symbol_min=None, symbol_max=None, DBG=False):
        """Encode quantized latent features with torchac arithmetic coding.

        In multi-stage mode this is causal: stage i is encoded only after
        stages 0..i-1 have been inserted into x_rec, so encoder and decoder use
        exactly the same previous-stage context.
        """
        if torchac is None:
            raise ImportError('torchac is not installed. Install it with: pip install torchac')
        if training:
            # Real entropy coding must be deterministic; do not use additive
            # uniform noise from the training quantizer.
            training = False
        if not require_quant:
            raise ValueError('Real entropy coding requires quantized integer symbols; set require_quant=True.')

        device = x.device
        N, C = x.F.size()
        ctxCat, mu_sigma = self._build_hyper_context(x, ctx=ctx)

        if self.adaptive_quant:
            qEnc_qDec = self.qpGenerator(ctxCat)
            qEnc_F, _ = torch.split(qEnc_qDec.F,
                                    [qEnc_qDec.size()[1] // 2, qEnc_qDec.size()[1] // 2], dim=1)
            x_enc_F = x.F * (1 + qEnc_F)
        else:
            x_enc_F = x.F
        x_hat_q_F = torch.round(x_enc_F).to(torch.float32)

        symbols_all = x_hat_q_F.detach().to(torch.long)
        if symbol_min is None:
            symbol_min = int(symbols_all.min().item()) if symbols_all.numel() else 0
        if symbol_max is None:
            symbol_max = int(symbols_all.max().item()) if symbols_all.numel() else 0
        if int(symbol_max) - int(symbol_min) + 1 > 32768:
            raise ValueError('torchac symbol alphabet is too large for int16 symbols.')

        stream_list = []

        if self.stage <= 1:
            mu_f, sigma_f = torch.split(mu_sigma.F,
                                        [mu_sigma.size()[1] // 2, mu_sigma.size()[1] // 2], dim=1)
            sigma_f = torch.exp(torch.clamp(sigma_f, max=10))
            stream_list.append(self._torchac_encode(x_hat_q_F, mu_f, sigma_f,
                                                    symbol_min, symbol_max,
                                                    gaussion=gaussion, DBG=DBG))
        else:
            mu_sigma_input = ME.SparseTensor(
                torch.cat([mu_sigma.F,
                           torch.zeros(N, self.hidden, dtype=torch.float32, device=device)], dim=1),
                coordinate_manager=mu_sigma.coordinate_manager,
                coordinate_map_key=mu_sigma.coordinate_map_key)
            x_rec = ME.SparseTensor(torch.zeros_like(x_hat_q_F),
                                    coordinate_manager=x.coordinate_manager,
                                    coordinate_map_key=x.coordinate_map_key)
            x_hat_q = ME.SparseTensor(x_hat_q_F,
                                      coordinate_manager=x.coordinate_manager,
                                      coordinate_map_key=x.coordinate_map_key)

            for stage_id in range(self.stage):
                coordinates = self._stage_coordinates(x, stage_id)
                if coordinates.size(0) == 0:
                    stream_list.append(b'')
                    continue

                # Only previous stages are available in x_rec here.
                stage_ctx = self.stageCtxExtract(x_rec, coordinates)
                stage_ctx = ME.SparseTensor(
                    torch.cat([torch.zeros(stage_ctx.size()[0], mu_sigma.size()[1],
                                           dtype=torch.float32, device=device),
                               stage_ctx.F], dim=1),
                    coordinates=coordinates,
                    coordinate_manager=mu_sigma_input.coordinate_manager)
                mu_sigma_input = sort_sparse_tensor(mu_sigma_input + stage_ctx, mu_sigma)
                stage_mu_sigma = self.ctxMergeFinal(mu_sigma_input) + mu_sigma
                stage_mu_sigma_F = self._features_at(stage_mu_sigma, coordinates)
                mu_f, sigma_f = torch.split(stage_mu_sigma_F,
                                            [stage_mu_sigma_F.size(1) // 2,
                                             stage_mu_sigma_F.size(1) // 2], dim=1)
                sigma_f = torch.exp(torch.clamp(sigma_f, max=10))

                stage_symbols_F = self._features_at(x_hat_q, coordinates)
                stream_list.append(self._torchac_encode(stage_symbols_F, mu_f, sigma_f,
                                                        symbol_min, symbol_max,
                                                        gaussion=gaussion, DBG=DBG))

                # Insert current true quantized symbols so later stages use the
                # same causal context as the decoder will have.
                stage_x_hat = ME.SparseTensor(stage_symbols_F, coordinates=coordinates,
                                              coordinate_manager=x_rec.coordinate_manager)
                x_rec = ME.SparseTensor((x_rec + stage_x_hat).F,
                                        coordinate_manager=x.coordinate_manager,
                                        coordinate_map_key=x.coordinate_map_key)
                assert (x_rec.C == x.C).all()

        bitstream = self._pack_hyperprior_bitstream(stream_list, symbol_min, symbol_max, N, C)

        if DBG:
            x_dec = self.decode(x, bitstream, gaussion=gaussion, ctx=ctx, bar=bar)
            if self.adaptive_quant:
                qEnc_qDec = self.qpGenerator(ctxCat)
                _, qDec_F = torch.split(qEnc_qDec.F,
                                        [qEnc_qDec.size()[1] // 2, qEnc_qDec.size()[1] // 2], dim=1)
                ref = x_hat_q_F * (1 + qDec_F)
            else:
                ref = x_hat_q_F
            assert torch.allclose(x_dec.F.detach().cpu(), ref.detach().cpu(), atol=1e-5)
        return bitstream

    @torch.no_grad()
    def decode(self, x_template, bitstream, gaussion=True, ctx=None, bar=False):
        """Decode a torchac HyperPriorMultiStage bitstream.

        x_template must have the same coordinates and coordinate manager layout
        as the encoded sparse tensor. Its feature values are ignored.
        """
        if torchac is None:
            raise ImportError('torchac is not installed. Install it with: pip install torchac')

        meta = self._unpack_hyperprior_bitstream(bitstream)
        N, C = x_template.F.size()
        if N != meta['num_points'] or C != meta['num_channels']:
            raise ValueError(
                f'Bitstream expects {(meta["num_points"], meta["num_channels"])} layout, '
                f'but x_template has {(N, C)}.'
            )

        device = x_template.device
        ctxCat, mu_sigma = self._build_hyper_context(x_template, ctx=ctx)
        if self.adaptive_quant:
            qEnc_qDec = self.qpGenerator(ctxCat)
            _, qDec_F = torch.split(qEnc_qDec.F,
                                    [qEnc_qDec.size()[1] // 2, qEnc_qDec.size()[1] // 2], dim=1)
        else:
            qDec_F = None

        if self.stage <= 1:
            mu_f, sigma_f = torch.split(mu_sigma.F,
                                        [mu_sigma.size()[1] // 2, mu_sigma.size()[1] // 2], dim=1)
            sigma_f = torch.exp(torch.clamp(sigma_f, max=10))
            x_hat_q_F = self._torchac_decode(meta['streams'][0], mu_f, sigma_f,
                                             meta['symbol_min'], meta['symbol_max'],
                                             gaussion=gaussion, device=device)
        else:
            mu_sigma_input = ME.SparseTensor(
                torch.cat([mu_sigma.F,
                           torch.zeros(N, self.hidden, dtype=torch.float32, device=device)], dim=1),
                coordinate_manager=mu_sigma.coordinate_manager,
                coordinate_map_key=mu_sigma.coordinate_map_key)
            x_rec = ME.SparseTensor(torch.zeros((N, C), dtype=torch.float32, device=device),
                                    coordinate_manager=x_template.coordinate_manager,
                                    coordinate_map_key=x_template.coordinate_map_key)

            for stage_id in range(self.stage):
                coordinates = self._stage_coordinates(x_template, stage_id)
                if coordinates.size(0) == 0:
                    continue

                # Decoder has only decoded stages 0..stage_id-1 in x_rec.
                stage_ctx = self.stageCtxExtract(x_rec, coordinates)
                stage_ctx = ME.SparseTensor(
                    torch.cat([torch.zeros(stage_ctx.size()[0], mu_sigma.size()[1],
                                           dtype=torch.float32, device=device),
                               stage_ctx.F], dim=1),
                    coordinates=coordinates,
                    coordinate_manager=mu_sigma_input.coordinate_manager)
                mu_sigma_input = sort_sparse_tensor(mu_sigma_input + stage_ctx, mu_sigma)
                stage_mu_sigma = self.ctxMergeFinal(mu_sigma_input) + mu_sigma
                stage_mu_sigma_F = self._features_at(stage_mu_sigma, coordinates)
                mu_f, sigma_f = torch.split(stage_mu_sigma_F,
                                            [stage_mu_sigma_F.size(1) // 2,
                                             stage_mu_sigma_F.size(1) // 2], dim=1)
                sigma_f = torch.exp(torch.clamp(sigma_f, max=10))

                stage_F = self._torchac_decode(meta['streams'][stage_id], mu_f, sigma_f,
                                               meta['symbol_min'], meta['symbol_max'],
                                               gaussion=gaussion, device=device)
                stage_x_hat = ME.SparseTensor(stage_F, coordinates=coordinates,
                                              coordinate_manager=x_rec.coordinate_manager)
                x_rec = ME.SparseTensor((x_rec + stage_x_hat).F,
                                        coordinate_manager=x_template.coordinate_manager,
                                        coordinate_map_key=x_template.coordinate_map_key)
                assert (x_rec.C == x_template.C).all()
            x_hat_q_F = x_rec.F

        x_hat_F = x_hat_q_F if qDec_F is None else x_hat_q_F * (1 + qDec_F)
        return ME.SparseTensor(x_hat_F,
                               coordinate_manager=x_template.coordinate_manager,
                               coordinate_map_key=x_template.coordinate_map_key)

    @torch.no_grad()
    def test_entropy_codec(self, x, gaussion=True, ctx=None):
        bitstream = self.encode(x, gaussion=gaussion, ctx=ctx, DBG=False)
        x_rec = self.decode(x, bitstream, gaussion=gaussion, ctx=ctx)
        return len(bitstream) * 8, x_rec

class ContextDetach(nn.Module):
    def __init__(self, input, xc_input, output=0, mode='0'):
        super(ContextDetach, self).__init__()
        self.conv = ME.MinkowskiConvolution(in_channels=input+xc_input, out_channels=input, kernel_size=3, stride=1, dimension=3)
        if mode!='0':
            self.compression = nn.Sequential(
                make_layer(ResNet, 1, input),
            )
            if mode=='1':
                self.compression.append(ME.MinkowskiConvolution(in_channels=input, out_channels=output, kernel_size=3, stride=1, dimension=3))
            elif mode=='2':
                self.compression.append(ME.MinkowskiConvolution(in_channels=input, out_channels=output, kernel_size=2, stride=2, dimension=3))
        else:
            self.compression = None
        self.mode = mode
    
    def forward(self, x, xc):
        if type(xc)==list:
            xcCat = ME.SparseTensor(torch.zeros_like(x.F[:, :0]), coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
            for c in xc:
                if c is None:
                    continue
                c = sparse_tensor_upsample(c, x)
                xcCat = ME.cat(xcCat, c)
            xc = xcCat
        else:
            xc = sparse_tensor_upsample(xc, x)
        x1 = ME.cat(x, xc)

        r = x-self.conv(x1)
        if self.mode!='0':
            r = self.compression(r)
        return r
    

class ContextRestore(nn.Module):
    def __init__(self, input, xc_input, xc_output=None, mode='0'):
        super(ContextRestore, self).__init__()
        if xc_output is None:
            xc_output = xc_input
        self.conv = ME.MinkowskiConvolution(in_channels=input+xc_input, out_channels=xc_output, kernel_size=3, stride=1, dimension=3)
        if mode!='0':
            self.decompression = nn.Sequential()
            if mode=='2':
                self.decompression.append(ME.MinkowskiConvolutionTranspose(in_channels=input, out_channels=input, kernel_size=2, stride=2, dimension=3, expand_coordinates=False))
            self.decompression.append(make_layer(ResNet, 1, input))
        if xc_output!=xc_input:
            self.adjust = nn.Sequential(
                ME.MinkowskiConvolution(in_channels=xc_input, out_channels=xc_input, kernel_size=1, stride=1, dimension=3),
                ME.MinkowskiReLU(),
                ME.MinkowskiConvolution(in_channels=xc_input, out_channels=xc_input, kernel_size=1, stride=1, dimension=3),
                ME.MinkowskiReLU(),
                ME.MinkowskiConvolution(in_channels=xc_input, out_channels=xc_output, kernel_size=1, stride=1, dimension=3),
            )
        self.mode = mode

    def forward(self, r, xc):
        if self.mode!='0':
            r = self.decompression(r)
        if type(xc)==list:
            xcCat = ME.SparseTensor(torch.zeros_like(r.F[:, :0]), coordinate_manager=r.coordinate_manager, coordinate_map_key=r.coordinate_map_key)
            for c in xc:
                if c is None:
                    continue
                c = sparse_tensor_upsample(c, r)
                xcCat = ME.cat(xcCat, c)
            xc = xcCat
        else:
            xc = sparse_tensor_upsample(xc, r)
        x = ME.cat(r, xc)
        x = self.conv(x)
        if x.size()[1]!=xc.size()[1]:
            xc = self.adjust(xc)
        x = xc+x
        return x

class Factorized(nn.Module):
    def __init__(self, input, dimension=2):
        super(Factorized, self).__init__()
        self.f1 = Bitparm(input, dimension=dimension)
        self.f2 = Bitparm(input, dimension=dimension)
        self.f3 = Bitparm(input, dimension=dimension)
        self.f4 = Bitparm(input, dimension=dimension, final=True)

    def get_p(self, x):
        y = self.f1(x)
        y = self.f2(y)
        y = self.f3(y)
        p = self.f4(y)
        return p

    def forward(self, x, training=True, requireQuant=True):
        if requireQuant:
            x_hat = quant(x.F, training=training, qs=1)
        else:
            x_hat = x
        p = self.get_p(x_hat+0.5)-self.get_p(x_hat-0.5)
        x_hat = ME.SparseTensor(x_hat, coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        b = torch.sum(torch.clamp(-1.0 * torch.log(p + 1e-10) / math.log(2.0), 0, 50))
        return {
            'x_hat': x_hat,
            'p_x': p,
            'b_x': b
        }
    

class DownSampler(torch.nn.Module):
    def __init__(self, in_channels=1, channels=32, out_channels=32, block_layers0=0, block_layers1=0, kernel_size=3,
                 block_type='conv', knn=16):
        super().__init__()
        if block_type == 'conv':
            self.block0 = torch.nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=in_channels, out_channels=channels,
                    kernel_size=3, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=True),
                ResNetBlock(block_layers=block_layers0, channels=channels, kernel_size=kernel_size))
        else:
            raise NotImplementedError

        self.conv = ME.MinkowskiConvolution(
            in_channels=channels, out_channels=channels,
            kernel_size=2, stride=2, bias=True, dimension=3)

        if block_type == 'conv':
            self.block1 = torch.nn.Sequential(
                ResNetBlock(block_layers=block_layers1, channels=channels, kernel_size=kernel_size),
                ME.MinkowskiConvolution(
                    in_channels=channels, out_channels=out_channels,
                    kernel_size=3, stride=1, bias=True, dimension=3))
        else:
            raise NotImplementedError

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.block0(x)
        out = self.relu(self.conv(out))
        out = self.block1(out)
        return out

class Upsampler(torch.nn.Module):
    def __init__(self, in_channels=1, channels=32, out_channels=32, block_layers0=0, block_layers1=0, kernel_size=3,
                 block_type='conv', knn=16):
        super().__init__()
        if block_type == 'conv':
            self.block0 = torch.nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=in_channels, out_channels=channels,
                    kernel_size=3, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=True),
                ResNetBlock(block_layers=block_layers0, channels=channels, kernel_size=kernel_size))
        else:
            raise NotImplementedError

        self.conv = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels, out_channels=channels,
            kernel_size=2, stride=2, bias=True, dimension=3)

        if block_type == 'conv':
            self.block1 = torch.nn.Sequential(
                ResNetBlock(block_layers=block_layers1, channels=channels, kernel_size=kernel_size),
                ME.MinkowskiConvolution(
                    in_channels=channels, out_channels=out_channels,
                    kernel_size=3, stride=1, bias=True, dimension=3))
        else:
            raise NotImplementedError

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x, coordinates=None):
        out = self.block0(x)
        out = self.relu(self.conv(out, coordinates=coordinates))
        out = ME.SparseTensor(out.F, coordinates=out.C, tensor_stride=x.tensor_stride[0]//2)
        out = self.block1(out)
        return out    


class Classifier(torch.nn.Module):
    """calculate the probability of voxel occupied.
        Todo: rename to MLP or FC or Linear Layer.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels, out_channels=channels,
            kernel_size=1, stride=1, bias=True, dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels, out_channels=channels,
            kernel_size=1, stride=1, bias=True, dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels, out_channels=1,
            kernel_size=1, stride=1, bias=True, dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.relu(self.conv1(out))
        out = self.conv2(out)

        return out


class UpSampler8StageP(torch.nn.Module):
    def __init__(self, in_channels=1, channels=32, kernel_size=5, block_layers=3, stage=8, block_type='conv', adaptor=False):
        super().__init__()
        self.stage = stage
        self.block_in = torch.nn.Sequential(
            self.make_block(in_channels=in_channels, channels=channels, out_channels=channels, 
                            kernel_size=1, block_layers=block_layers, block_type='conv'),
            ME.MinkowskiConvolutionTranspose(in_channels=channels, out_channels=channels, kernel_size=[2,2,2], 
                                            stride=[2,2,2], expand_coordinates=True, bias=True, dimension=3))
        self.block_list = torch.nn.ModuleList([
            self.make_block(in_channels=channels, channels=channels, out_channels=channels, 
                            kernel_size=kernel_size, block_layers=block_layers, block_type='conv_mask') for i in range(self.stage)])
        self.classifier = Classifier(channels)
        self.pruning = ME.MinkowskiPruning()

        self.unpooling = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, expand_coordinates=True, dimension=3)
        self.unpooling_skip = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, expand_coordinates=False, dimension=3)
        self.sumpooling = ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=3)
        self.contextConv = ME.MinkowskiConvolution(in_channels=channels+channels, out_channels=channels, kernel_size=3, stride=1, bias=True, dimension=3)
        if adaptor:
            self.adaptor = nn.Sequential(
                ME.MinkowskiConvolution(in_channels=channels, out_channels=in_channels, kernel_size=1, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=True, dimension=3),
            )
        else:
            self.adaptor = None
        
    
    def make_block(self, in_channels=32, channels=32, out_channels=32, kernel_size=3, block_layers=3, knn=16, block_type='conv'):
        if block_type=='conv_mask': 
            return torch.nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=in_channels, out_channels=channels,
                    kernel_size=kernel_size, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=True),
                ResNetBlock(block_layers=block_layers, channels=channels, kernel_size=kernel_size, block_type='inception_mask'),
                ME.MinkowskiConvolution(
                    in_channels=channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=1, bias=True, dimension=3))

        if block_type=='conv': 
            return torch.nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=in_channels, out_channels=channels,
                    kernel_size=kernel_size, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=True),
                ResNetBlock(block_layers=block_layers, channels=channels, kernel_size=kernel_size),
                ME.MinkowskiConvolution(
                    in_channels=channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=1, bias=True, dimension=3))
        raise NotImplementedError

    
    def split(self, x):
        device = x.device
        slice_list = []
        octant = torch.sum((torch.div(x.C[:, 1:], x.tensor_stride[0], rounding_mode='floor')
                            %torch.tensor(2).to(device))
                           * torch.tensor([1, 2, 4]).to(device), axis=1)
        if self.stage==8:
            offset_list = [[0],[7],[1],[6],[5],[2],[3],[4]]
        elif self.stage==3:
            offset_list = [[0,1], [6,7], [2,3,4,5]]
        elif self.stage==2:
            offset_list = [[0,1,6,7], [2,3,4,5]]
        for _, offsets in enumerate(offset_list):
            mask = torch.sum(torch.stack([octant==a for a in offsets]), axis=0).bool()
            curr_slice = self.pruning(x, mask)
            slice_list.append(curr_slice)

        return slice_list

    def concat(self, x0, x1):
        feats = torch.cat([x0.F, x1.F], dim=0)
        coords=torch.cat([x0.C, x1.C], dim=0)
        out = create_new_sparse_tensor(
            features=feats, 
            coordinates=coords, 
            tensor_stride=x0.tensor_stride, 
            dimension=x0.D, device=x0.device)

        return out

    def basic_module(self, block, classifier, x_in, siblings, ground_truth, prob=None, prob_ste=None, siblings_prob=None):
        # concat  x_in and siblings
        prob_flag = (prob is not None) and (prob_ste is not None)
        if siblings is None:
            inputs = x_in
        else: 
            inputs = self.concat(siblings, x_in)
        # & extract feature
        if siblings_prob is not None:
            inputs_prob_ste = self.concat(siblings_prob, prob_ste)
        elif prob_ste is not None:
            inputs_prob_ste = prob_ste
        else:
            inputs_prob_ste = None
        if inputs_prob_ste is not None:
            inputs_prob_ste = sort_sparse_tensor(inputs_prob_ste, inputs)

        out = block[0](inputs)
        out = block[1](out)
        out, mask = block[2](out, mask=inputs_prob_ste, withmask=True)

        # out = block[2](out)
        if inputs_prob_ste is not None:
            out = out * inputs_prob_ste
        out = block[3](out)
        if siblings is None:
            x_out = out
        else:
            mask_in = isin(out.C, x_in.C)
            x_out = self.pruning(out, mask_in.to(out.device))
        out_cls = classifier(x_out)
        mask_out = isin(out.C, ground_truth.C).to(out.device)
        out = self.pruning(out, mask_out)
        if prob is not None:
            out_mask = self.pruning(inputs_prob_ste, mask_out)
        else:
            out_mask = None

        return out, out_cls, out_mask


    def forward(self, x_low, x_high, training, residual=None, context=False, prob=None, prob_ste=None):
        if prob is not None:
            prob = sort_sparse_tensor(prob, x_low)
        x_low = self.block_in(x_low)
        if prob is not None:
            upsample = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, expand_coordinates=True, dimension=3)
            prob_up = upsample(prob)
            prob_ste_up = upsample(prob_ste)
            prob_list = self.split(prob_up)
            prob_ste_list = self.split(prob_ste_up)
            # P-SOPA implementation step 1, multiply input by mask. Following context convolutions also need to be masked before implementation.
            x_low = x_low* prob_ste_up
        else:
            prob_list = [None]*8
            prob_ste_list = [None]*8
            prob_up = None
            prob_ste_up = None
        ### todo: should add prediction for residual or not?
        slice_list = self.split(x_low)

        out_cls_list, target_list = [], []
        out = None
        sib_prob = None
        for i, (block, curr_slice) in enumerate(zip(self.block_list, slice_list)):
            out, out_cls, sib_prob = self.basic_module(block=block, classifier=self.classifier,
                                             x_in=curr_slice, siblings=out, ground_truth=x_high, prob=prob_list[i], prob_ste=prob_ste_list[i], siblings_prob=sib_prob)
            # print('DBG!!!', out.shape, out_cls.shape)
            out_cls_list.append(out_cls)
            target_list.append(isin(out_cls.C, x_high.C))
        if residual is not None:
            residual = sort_sparse_tensor(residual, out)
            assert (residual.C==out.C).all()
            if context:
                ctx = ME.cat(residual, out)
                out = out+self.contextConv(ctx)
            else:
                out = out+residual

        if self.adaptor is not None:
            out = self.adaptor(out)

        return {'out': out, 'target_list': target_list,
                'out_cls_list': out_cls_list, 'prob_up': prob_up, 'prob_ste_up':prob_ste_up}

    # @torch.no_grad()
    # def encode(self, x_low, x_high, DBG=True):
    #     #### TBD ####
    #     #### If needed, can use the same AE/AD as SparsePCGC. ####
    #     #### Note that our P-SOPA is only a training trick, for real compression, it's the same as SOPA. ####


    # @torch.no_grad()
    # def decode(self, x_low, bitstream):
    #     #### TBD ####

class PoolingTransposeWithRefine(nn.Module):
    def __init__(self, channels, kernel_size=8, stride=8, dimension=3):
        super().__init__()

        # 保留原来的行为
        self.up = ME.MinkowskiPoolingTranspose(
            kernel_size=kernel_size,
            stride=stride,
            dimension=dimension,
            expand_coordinates=True
        )

        # self.up = ME.MinkowskiPoolingTranspose(kernel_size=8, stride=8, dimension=3, expand_coordinates=True)

        # 3×3×3 refine
        self.refine = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            stride=1,
            bias=False,
            dimension=dimension
        )

        self._init_identity_refine()

    def _init_identity_refine(self):
        with torch.no_grad():
            weight = self.refine.kernel  # (K, C_in, C_out)，例如 (27, 3, 3)
            weight.zero_()

            K, C_in, C_out = weight.shape
            center_idx = K // 2          # 3×3×3 -> index 13

            C = min(C_in, C_out)
            for c in range(C):
                weight[center_idx, c, c] = 1.0   # 中心位置通道对角线 = 1

    def forward(self, x, coordinates=None):
        if coordinates is None:
            y = self.up(x)
        else:
            y = self.up(x, coordinates)
        y1 = self.refine(y)
        return y1