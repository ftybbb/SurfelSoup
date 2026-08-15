import argparse
import collections
import importlib
import os
import sys
import open3d
import math
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, os.pardir))
import torch
import glob
import numpy as np
from dataset import rgb_to_yuv, yuv2rgb
import MinkowskiEngine as ME
from pytorch3d.ops import knn_points
from models import *

def parse_args():
    #### hint: most parameters are for training, but we still keep them, such that you just need to copy the training settings for evaluation here.
    parser = argparse.ArgumentParser(description='SurfelSoup')
    parser.add_argument('--out_dir', default='/scratch/tf2387/geo_result', type=str, help='output directory [default: /scratch/tf2387/geo_result]')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training [default: 4]')
    parser.add_argument('--actual_batch_size', type=int, default=1, help='actual batch size when forwarding and backwarding [default: 1]')
    parser.add_argument('--model', default='octsurfel', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training [default: 100]')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training [default: 0.0001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--lamb', default=1.0, type=float, help='loss=lambda*D+R')
    parser.add_argument('--pretrained', default='', type=str, help='load model weights here')
    parser.add_argument('--frozen_modules', nargs='+')
    parser.add_argument('--exp_name', default='', type=str, help='directory to store results')
    parser.add_argument('--channels', default=32, type=int, help='latent channels')
    parser.add_argument('--activation', default='relu', type=str)
    parser.add_argument('--dataset_dir', default=['/scratch/tf2387/datasets/8iVFB/longdress/Ply', '/scratch/tf2387/datasets/8iVFB/redandblack/Ply', '/scratch/tf2387/datasets/8iVFB/soldier/Ply', '/scratch/tf2387/datasets/8iVFB/loot/Ply'], type=str, nargs='+')  # '/workspace/xrx/dataset/cfp/train/dense_dynamic/8i/8iVFBv2'
    parser.add_argument('--scaling_factor', default=1, type=int)
    parser.add_argument("--augment", default=False, type=bool, help="data augment.") 
    parser.add_argument("--hyper", default=1, type=int, help="use hyperprior if set to 1, factorized model if set to 0.")  
    parser.add_argument("--max_iter", default=999999, type=int, help="maximum number of iterations per epoch.")
    parser.add_argument('--val_num', default=0.05, type=float, help='the number of frames used for validation, can be a float number between 0 and 1, which means the ratio of validation frames, or an integer number which means the exact number of validation frames')
    parser.add_argument('--ignore_module_load', default=[], type=str, nargs='+')
    parser.add_argument('--layers', default=5, type=int)
    parser.add_argument('--random', default=0, type=float)
    parser.add_argument('--adaptive_quant', default=0, type=int)
    parser.add_argument('--bar', default=0, type=int, help='set to 1 to use STE for quantization, set to 0 to add noise in replacement of quantization')
    parser.add_argument('--save_every', default=10, type=int, help='save frequency (unit: epoch)')
    parser.add_argument('--octree_end', default=1, type=int, help='any nodes in layers lower than octree_end will be surfel nodes')
    parser.add_argument('--surfel_start', default=2, type=int, help='any nodes in layers higher than surfel_start will be octree nodes')
    parser.add_argument('--selection_mode', default='gumbel', type=str, help='selection mode for decision module, can be fixed or gumbel')
    parser.add_argument('--from_step', default=0, type=int, help='from which step to start training, this relates to the temperature of gumbel softmax')
    parser.add_argument('--step_size', default=10, type=int, help='step size for learning rate decay')
    return parser.parse_args()

args = parse_args()
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

models_and_ckpts = [
    # [model_name, checkpoint_path, use_super_resolution (for extreme low bit rate you need this), rho (reconstruct rho*N points)]
    ['SurfelSoup_v1', './ckpts/g15_7.pth', False, 1.0],
    ['SurfelSoup_v1', './ckpts/g10_7.pth', False, 1.0],
    ['SurfelSoup_v1', './ckpts/g08_7.pth', False, 1.0],
    ['SurfelSoup_v1', './ckpts/g05_7.pth', False, 1.0],
    ['SurfelSoup_v1', './ckpts/g03_7.pth', False, 1.0],
    ['SurfelSoup_v1', './ckpts/g01_7.pth', False, 1.0]
]

test_sequences = [
    ['/scratch/tf2387/datasets/rwtt1/point_clouds3/vox10/RWT_156_vox10.ply', 1023],
]

with torch.no_grad():
    for sequence, resolution in test_sequences:
        if os.path.isdir(sequence):
            files = sorted(glob.glob(os.path.join(sequence, '**', '*.ply'), recursive=True))
        else:
            files = [sequence]
        files = [os.path.join(sequence, f) for f in files]
        res_bpp = np.asarray([0.0 for i in range(len(models_and_ckpts))])
        res_psnr = np.asarray([0.0 for i in range(len(models_and_ckpts))])
        res_psnr_d2 = np.asarray([0.0 for i in range(len(models_and_ckpts))])
        for k, model_and_ckpt_and_sr in enumerate(models_and_ckpts):
            model_name, ckpt, sr, rho = model_and_ckpt_and_sr
            MODEL = importlib.import_module(model_name)
            device = torch.device('cuda:0')
            # load model
            model = MODEL.get_model(channels=args.channels, hyper=args.hyper, bar=args.bar, octree_end=args.octree_end, surfel_start=args.surfel_start, adaptive_quant=args.adaptive_quant, selection_mode=args.selection_mode).to(device)  # todo
            checkpoint = torch.load(ckpt)
            old_params = model.state_dict()
            new_state_dict = collections.OrderedDict()
            for k0, v in checkpoint.items():
                k1 = k0.replace('module.', '')
                if k1 not in old_params:
                    print(k1, 'not exist')
                    continue
                if old_params[k1].size()!=v.size():
                    print(k1, old_params[k1].size(), v.size(), 'not matched')
                    continue
                new_state_dict[k1] = v
            old_params.update(new_state_dict)
            model.load_state_dict(old_params)
            model.eval()

            # start traversing
            for i, file in enumerate(files):
                file_prefix = file.replace('.ply', '').split('/')[-1]
                print(file_prefix)
                pcOpen3d = open3d.io.read_point_cloud(file)
                coordinates = torch.tensor(np.asarray(pcOpen3d.points), dtype=torch.int32, device=device)
                coordinates = torch.cat([torch.zeros_like(coordinates[:, :1]), coordinates], dim=1)
                colors = np.asanyarray(pcOpen3d.colors)
                occupancies = np.ones_like(colors[:, :1])
                if not pcOpen3d.has_normals():
                    pcOpen3d.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamKNN(knn=16))
                normals = np.asarray(pcOpen3d.normals)
                normals = torch.tensor(normals, dtype=torch.float32, device=device)
                colors = torch.tensor(colors, dtype=torch.float32, device=device)
                occupancies = torch.tensor(occupancies, dtype=torch.float32, device=device)
                f2 = ME.SparseTensor(colors, coordinates=coordinates, tensor_stride=1, device=device)
                f2_geo = ME.SparseTensor(occupancies, coordinates=coordinates, tensor_stride=1, device=device)
                num_points = f2.size()[0]
                out = model.encode(f2_geo, f2_geo, device=device, keep_adaptive=False, step=10000)
                bpp = out['bpp']*f2.size()[0]/num_points
                topkplus1 = out['topkplus1']
                bpp = bpp.item()
                coordinates_all = out['coordinates']
                occupancies_all = out['occupancies'].squeeze()
                _, idx = torch.topk(occupancies_all, k=int(f2.size()[0]*rho))
                coordinates_rec = coordinates_all[idx][:, 1:]

                if topkplus1 is not None:
                    coordinates_rec = torch.unique(torch.cat([coordinates_rec, topkplus1], dim=0), dim=0)

                # recolor
                knn_dist, knn_idx, _ = knn_points(coordinates_rec.unsqueeze(0).to(torch.float32), coordinates[:, 1:].unsqueeze(0).to(torch.float32), K=3)
                knn_dist = knn_dist.squeeze(0)
                knn_colors = index_points(colors.unsqueeze(0), knn_idx)
                knn_colors = knn_colors.squeeze(0)
                inverse_dist = 1/(knn_dist+0.0000001)
                weight = inverse_dist / inverse_dist.sum(dim=-1, keepdim=True)
                knn_colors = (knn_colors * weight.unsqueeze(-1)).sum(dim=-2)
                pcOpen3d_rec = open3d.geometry.PointCloud()
                pcOpen3d_rec.points = open3d.utility.Vector3dVector(coordinates_rec.detach().cpu().numpy())
                pcOpen3d_rec.colors = open3d.utility.Vector3dVector(knn_colors.detach().cpu().numpy())
                open3d.io.write_point_cloud(os.path.join(args.out_dir, 'rec.ply'), pcOpen3d_rec)

                # compute psnr
                d1, d2, _ = PSNR(f2.C[:, 1:].unsqueeze(0), coordinates_rec.unsqueeze(0), n1=normals.unsqueeze(0), peak_value=resolution)
                print('####', sequence, files[i], k, bpp, d1, d2)
                res_bpp[k] += bpp
                res_psnr[k] += d1
                res_psnr_d2[k] += d2
            res_bpp[k] = res_bpp[k]/len(files)
            res_psnr[k] = res_psnr[k]/len(files)
            res_psnr_d2[k] = res_psnr_d2[k]/len(files)

        print('@@@@', sequence, list(res_bpp), list(res_psnr), list(res_psnr_d2))
