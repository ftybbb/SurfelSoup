import numpy as np
import open3d
import torch
import torch.utils.data as data
from os.path import join
import os
import MinkowskiEngine as ME
import random
import glob
import h5py

'''
支持intra+inter
输入filesdir
eval不取最后0.05，换成等间隔均匀选择，和nju对齐
'''

class StaticDataset(data.Dataset):
    def __init__(self, seqs_list, type='train', scaling_factor=1, attribute=False, yuv=0, crop=-1, color_aug=0):
        self.scaling_factor = scaling_factor
        self.format = os.path.basename(seqs_list[0]).split('.')[-1]
        self.lookup = seqs_list
        self.attribute = attribute
        self.yuv = yuv
        if not isinstance(crop, int) or crop == 0 or crop < -1:
            raise ValueError(f'crop must be -1 or a positive integer, got {crop!r}')
        self.crop = crop
        self.type = type
        self.color_aug = color_aug
        if yuv==1:
            print('YUV input')

    def __getitem__(self, item):
        file_dir = self.lookup[item]
        if self.format == 'npy':
            points = np.load(file_dir)
        elif self.format == 'ply':
            point_cloud = open3d.io.read_point_cloud(file_dir)
            points = np.asarray(point_cloud.points)
            colors = np.asanyarray(point_cloud.colors)
            if self.yuv!=0:
                colors = rgb_to_yuv(colors)
            colors = torch.tensor(colors)
        elif self.format == 'h5':
            f = h5py.File(file_dir, 'r')
            raise NotImplementedError('h5 format not implemented yet')
        colors = torch.tensor(colors)
        coordinates = torch.tensor(points[:, :3])
        if self.scaling_factor != 1:
            coordinates = torch.unique(
                torch.floor(coordinates / self.scaling_factor), dim=0
            )

        if self.crop != -1 and coordinates.size(0) > self.crop:
            # Pick an existing point as the crop centre, so the crop is never empty.
            center = coordinates[random.randrange(coordinates.size(0))]

            # Binary-search the smallest cubic window containing at least 70% of
            # the requested points. Unlike the old additive scale update, this is
            # bounded and does not assume that the coordinate range is 1024.
            min_points = max(1, int(np.ceil(self.crop * 0.7)))
            distances = torch.amax(torch.abs(coordinates - center), dim=1)
            low, high = 0.0, distances.max().item()
            mask = distances <= high
            for _ in range(24):
                window_radius = (low + high) / 2
                candidate_mask = distances <= window_radius
                if candidate_mask.sum().item() >= min_points:
                    high = window_radius
                    mask = candidate_mask
                else:
                    low = window_radius

            selected = torch.nonzero(mask, as_tuple=False).squeeze(1)

            # Boundary points can make the cube exceed crop; cap it explicitly
            # while applying exactly the same indices to coordinates and colours.
            if selected.numel() > self.crop:
                selected = selected[torch.randperm(selected.numel())[:self.crop]]
            coordinates, colors = coordinates[selected], colors[selected]


        if self.color_aug!=0 and self.type=='train':
            factor = [random.uniform(0, self.color_aug)-self.color_aug/2 for i in range(3)]
            colors[:, 0] += factor[0]
            colors[:, 1] += factor[1]
            colors[:, 2] += factor[2]
            colors = torch.clamp(colors, min=0, max=1)
        features = (
            colors
            if self.attribute
            else torch.ones_like(coordinates[:, :1])
        )
        return coordinates, features

    def __len__(self):
        return len(self.lookup)

def static_collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError('No data in the batch')

    coordinates, colors = list(zip(*list_data))

    coordinates_batch = ME.utils.batched_coordinates(coordinates)
    colors_batch = torch.vstack(colors).float()
    return coordinates_batch, colors_batch


def get_seqs_list(filedirs):
    filedirs = sorted(filedirs)
    if 'ShapeNet' in filedirs[0]:
        filedirs = sorted(filedirs, key=lambda x: int(os.path.basename(x).split('.')[0]))
    filedirs = [f for f in filedirs if f[-5]!='_']
    seqs_list = []

    for idx in range(len(filedirs)):
        if idx == 0: continue
        curr_frame = filedirs[idx]
        ref_frame = filedirs[idx - 1]

        Idx0 = os.path.split(curr_frame)[-1].split('.')[0].split('_')[-1]
        Idx1 = os.path.split(ref_frame)[-1].split('.')[0].split('_')[-1]

        if 'ShapeNet' in curr_frame:
            Idx0 = int(os.path.basename(curr_frame).split('.')[0])
            Idx1 = int(os.path.basename(ref_frame).split('.')[0])

        try:
            Idx0 = int(Idx0)
            Idx1 = int(Idx1)
        except (ValueError) as e:
            Idx0 = int(Idx0.split('-')[-1])
            Idx1 = int(Idx1.split('-')[-1])

        if 'Owlii' in curr_frame and Idx0>=900:
            continue
        if (Idx0 - Idx1 != 1): continue
        seqs_list.append([ref_frame, curr_frame])

    return seqs_list

def rgb_to_yuv(x, dim=-1):
    r, g, b = x[:, :1], x[:, 1:2], x[:, 2:3]
    Kr, Kg, Kb = (0.2126, 0.7152, 0.0722)
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    if isinstance(x, np.ndarray):
        ycbcr = np.concatenate([y, cb, cr], axis=-1)
    else:
        ycbcr = torch.cat([y, cb, cr], dim=-1)
    return ycbcr

def yuv2rgb(x):
    y, cb, cr = x[:, :1], x[:, 1:2], x[:, 2:3]
    Kr, Kg, Kb = (0.2126, 0.7152, 0.0722)
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    if isinstance(x, np.ndarray):
        rgb = np.concatenate((r, g, b), axis=-1)
    else:
        rgb = torch.cat((r, g, b), dim=-1)
    return rgb
