import os
import re
import glob
import argparse
import collections
import numpy as np
import torch
import open3d as o3d
from plyfile import PlyData, PlyElement
from PIL import Image

# 完全关闭像素上限检查
Image.MAX_IMAGE_PIXELS = None

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
import MinkowskiEngine as ME

from concurrent.futures import ThreadPoolExecutor


###############################################
# utils
###############################################

def extract_frame_no(path):
    m = re.search(r"RWT[_\-]?(\d+)", path)
    if m:
        return int(m.group(1))

    m = re.search(r"scene(\d+)_(\d+)", path)
    if m:
        return int(m.group(1))

    return None

import torch
import math
import torch.nn as nn

def gaussian_kernel_3d(kernel_size=3, sigma=0.8):
    ax = torch.arange(-(kernel_size//2), kernel_size//2 + 1)
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')

    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    return kernel

class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=3, sigma=0.8, dimension=3):
        super().__init__()

        self.channels = 3

        self.conv = ME.MinkowskiConvolution(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            bias=False,
            dimension=dimension
        )

        self._init_gaussian(kernel_size, sigma)

        # 不训练
        for p in self.conv.parameters():
            p.requires_grad = False


    def _init_gaussian(self, kernel_size, sigma):
        kernel = gaussian_kernel_3d(kernel_size, sigma)

        kernel = kernel.view(-1)

        weight = torch.zeros(kernel.numel(), self.channels, self.channels)

        for c in range(self.channels):
            weight[:, c, c] = kernel

        self.conv.kernel.data = weight


    def forward(self, x):
        return self.conv(x)

###############################################
# ply io
###############################################

def write_ply_xyz_rgb(path, xyz, rgb):

    vertex = np.empty(
        xyz.shape[0],
        dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('red', 'u1'),
            ('green', 'u1'),
            ('blue', 'u1'),
        ],
    )

    vertex['x'] = xyz[:, 0]
    vertex['y'] = xyz[:, 1]
    vertex['z'] = xyz[:, 2]

    vertex['red'] = rgb[:, 0]
    vertex['green'] = rgb[:, 1]
    vertex['blue'] = rgb[:, 2]

    PlyData([PlyElement.describe(vertex, "vertex")], text=True).write(path)


def read_ply_xyz_rgb(path):

    v = PlyData.read(path)["vertex"].data

    xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)

    rgb = np.vstack([
        v['red'],
        v['green'],
        v['blue']
    ]).T.astype(np.uint8)

    return xyz, rgb


###############################################
# normalize
###############################################

def normalize_points(xyz, target_half=100):

    center = xyz.mean(axis=0)
    xyz = xyz - center

    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)

    half = np.max(np.maximum(np.abs(mn), np.abs(mx)))

    scale = target_half / half

    return xyz * scale


###############################################
# quantize
###############################################

def quantize_points(xyz, qp):

    min_xyz = xyz.min()
    max_xyz = xyz.max()

    span = np.maximum(max_xyz - min_xyz, 1e-6)

    grid = (1 << qp) - 1
    grid = 1200

    coords = np.rint((xyz - min_xyz) / span * grid).astype(np.int32)

    print('point span:', span, 'min_xyz:', min_xyz)

    coords = np.clip(coords, 0, grid)

    return coords


###############################################
# Minkowski unique
###############################################

def me_unique(coords_int: np.ndarray, feats_rgb: np.ndarray):
    """
    Same signature / returns as before:
      input:
        coords_int: (N,3) int32 voxel coords
        feats_rgb:  (N,3) uint8 or float (0..255)
      output:
        unique_coords: (M,3) int32
        unique_rgb_u8: (M,3) uint8  (voxel-mean color)
    """
    # Make contiguous (ME/torch ops like this)
    coords_np = np.ascontiguousarray(coords_int.astype(np.int32, copy=False))
    feats_np = np.ascontiguousarray(feats_rgb)

    coords = torch.from_numpy(coords_np).int().contiguous()          # (N,3)
    feats = torch.from_numpy(feats_np).float().contiguous()          # (N,3), color in 0..255

    # 1) Unique voxel coords + inverse map
    #    NOTE: torch.unique may return coords in sorted order (usually OK).
    unique_coords, inverse = torch.unique(coords, dim=0, return_inverse=True)

    # 2) Accumulate color sums per voxel
    M = unique_coords.shape[0]
    sums = torch.zeros((M, feats.shape[1]), dtype=torch.float32, device=feats.device)
    counts = torch.zeros((M,), dtype=torch.float32, device=feats.device)

    sums.index_add_(0, inverse, feats)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))

    # 3) Mean + to uint8
    mean = sums / counts.unsqueeze(1).clamp_min(1.0)
    unique_rgb_u8 = torch.clamp(torch.round(mean), 0, 255).to(torch.uint8).cpu().numpy()

    unique_coords_int = unique_coords.cpu().numpy().astype(np.int32)
    return unique_coords_int, unique_rgb_u8


###############################################
# color smoothing
###############################################

def smooth_colors_knn_mt(xyz, rgb, k=24, sigma_s=2.0, sigma_c=25.0, threads=8):

    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.float32)

    pts = o3d.core.Tensor(xyz)

    nns = o3d.core.nns.NearestNeighborSearch(pts)
    nns.knn_index()

    idx, dist2 = nns.knn_search(pts, k)

    idx = idx.numpy()
    dist2 = dist2.numpy()

    ws = np.exp(-dist2 / (2 * sigma_s * sigma_s))

    out = np.zeros_like(rgb)

    def process_range(i0, i1):

        neigh = idx[i0:i1]
        neigh_rgb = rgb[neigh]

        center = rgb[i0:i1][:, None, :]

        dc = np.sum((neigh_rgb - center) ** 2, axis=2)

        wc = np.exp(-dc / (2 * sigma_c * sigma_c))

        w = ws[i0:i1] * wc

        wsum = w.sum(axis=1, keepdims=True) + 1e-6

        sm = (neigh_rgb * w[:, :, None]).sum(axis=1) / wsum

        # out[i0:i1] = sm

        alpha = 0.5
        out[i0:i1] = (1.0 - alpha) * rgb[i0:i1] + alpha * sm

    N = xyz.shape[0]

    step = max(N // threads, 1)

    jobs = []

    for i in range(0, N, step):

        jobs.append((i, min(i + step, N)))

    with ThreadPoolExecutor(max_workers=threads) as ex:

        for i0, i1 in jobs:
            ex.submit(process_range, i0, i1)

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


###############################################
# sample step
###############################################

def sample(args):

    objs = sorted(
        glob.glob(os.path.join(args.mesh_dir, "*", "*.obj"))
        + glob.glob(os.path.join(args.mesh_dir, "*", "*.OBJ"))
    )

    frame_map = {}

    for p in objs:

        f = extract_frame_no(p)

        if f is not None:
            frame_map[f] = p

    ordered = collections.OrderedDict(sorted(frame_map.items()))

    frames = list(ordered.keys())
    files = list(ordered.values())

    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    base = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "1"))

    idx = task - base

    obj = files[idx]
    frame = frames[idx]

    print("Sampling", obj)

    out_dir = os.path.join(args.point_dir, args.sample_folder)
    os.makedirs(out_dir, exist_ok=True)

    out = os.path.join(out_dir, f"RWT_{frame:03d}.ply")

    mesh = load_objs_as_meshes([obj])

    xyz, rgb = sample_points_from_meshes(
        mesh,
        num_samples=args.sample_pts,
        return_textures=True
    )

    xyz = xyz[0].cpu().numpy()
    rgb = rgb[0].cpu().numpy()

    # xyz = normalize_points(xyz)

    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    write_ply_xyz_rgb(out, xyz, rgb)

    print("Saved", out)


###############################################
# quantize step
###############################################

def quantize(args):

    plys = sorted(glob.glob(os.path.join(args.point_dir, args.sample_folder, "*.ply")))

    frame_map = {}

    for p in plys:

        f = extract_frame_no(p)

        if f is not None:
            frame_map[f] = p

    ordered = collections.OrderedDict(sorted(frame_map.items()))

    frames = list(ordered.keys())
    files = list(ordered.values())

    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    base = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "1"))

    idx = task - base

    ply = files[idx]
    frame = frames[idx]

    print("Quantizing", ply)

    out_dir = os.path.join(args.point_dir, f"vox{args.qp}")
    os.makedirs(out_dir, exist_ok=True)

    out = os.path.join(out_dir, f"RWT_{frame:03d}_vox{args.qp}.ply")

    xyz, rgb = read_ply_xyz_rgb(ply)

    coords = quantize_points(xyz, args.qp)

    coords, rgb = me_unique(coords, rgb)

    xyz = coords.astype(np.float32)

    if args.color_smooth:

        # rgb = smooth_colors_knn_mt(
        #     xyz,
        #     rgb,
        #     k=args.color_k,
        #     sigma_s=args.sigma_s,
        #     sigma_c=args.sigma_c,
        #     threads=args.threads
        # )

        coords_np = coords.copy()
        coords = torch.from_numpy(coords).int()
        rgb = torch.from_numpy(rgb).float()

        coords = ME.utils.batched_coordinates([coords])

        st = ME.SparseTensor(features=rgb, coordinates=coords)

        gauss = GaussianFilter(kernel_size=3, sigma=0.2)

        # RGB filtering
        rgb_out = gauss(st)

        # normalization
        ones_feat = torch.ones_like(st.F)  # shape (N, C)
        ones = ME.SparseTensor(
            features=ones_feat,
            coordinate_manager=st.coordinate_manager,
            coordinate_map_key=st.coordinate_map_key
        )
        norm = gauss(ones)
        rgb_norm = rgb_out.F / (norm.F + 1e-6)

        rgb = torch.clamp(rgb_norm, 0, 255).byte().cpu().numpy()
    write_ply_xyz_rgb(out, xyz, rgb)

    print("Saved", out)


###############################################
# main
###############################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--step", default="quantize", choices=["sample", "quantize"])

    parser.add_argument("--mesh_dir", default="/scratch/tf2387/datasets/rwtt1")
    parser.add_argument("--point_dir", default="/scratch/tf2387/datasets/rwtt1/")

    parser.add_argument("--sample_folder", default="point_clouds5")

    parser.add_argument("--sample_pts", type=int, default=8000000)

    parser.add_argument("--qp", type=int, default=10)

    parser.add_argument("--color_smooth", type=int, default=1)
    parser.add_argument("--color_k", type=int, default=20)
    parser.add_argument("--sigma_s", type=float, default=2.0)
    parser.add_argument("--sigma_c", type=float, default=20.0)

    parser.add_argument("--threads", type=int, default=16)

    args = parser.parse_args()

    if args.step == "sample":
        sample(args)

    elif args.step == "quantize":
        quantize(args)