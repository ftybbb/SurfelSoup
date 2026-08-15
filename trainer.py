'''
在trainer_v1_inter.py的基础上
考虑上采样调用1/8stage的情况，对计算distortion做修改
'''
import argparse


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser('SurfelTrain')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training [default: 4]')
    parser.add_argument('--actual_batch_size', type=int, default=1, help='actual batch size when forwarding and backwarding [default: 1]')
    parser.add_argument('--model', default='octsurfel', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training [default: 100]')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training [default: 0.0001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--lamb', default=1.0, type=float, help='loss=lambda*D+R')
    parser.add_argument('--pretrained', default='', type=str, help='load model weights here')
    parser.add_argument('--frozen_modules', nargs='+')
    parser.add_argument('--root_dir', default='/scratch/tf2387/log', type=str, help='root directory of dataset')
    parser.add_argument('--exp_name', default='', type=str, help='directory to store results')
    parser.add_argument('--channels', default=64, type=int, help='latent channels')
    parser.add_argument('--activation', default='relu', type=str)
    parser.add_argument('--dataset_dir', default=['/scratch/tf2387/datasets/8iVFB/longdress/Ply', '/scratch/tf2387/datasets/8iVFB/redandblack/Ply', '/scratch/tf2387/datasets/8iVFB/soldier/Ply', '/scratch/tf2387/datasets/8iVFB/loot/Ply'], type=str, nargs='+')  # '/workspace/xrx/dataset/cfp/train/dense_dynamic/8i/8iVFBv2'
    parser.add_argument('--scaling_factor', default=1, type=int)
    parser.add_argument("--augment", default=False, type=bool, help="data augment.") 
    parser.add_argument("--hyper", default=0, type=int, help="use hyperprior if set to 1, factorized model if set to 0.")  
    parser.add_argument("--max_iter", default=999999, type=int, help="maximum number of iterations per epoch.")
    parser.add_argument('--val_num', default=0.05, type=float, help='the number of frames used for validation, can be a float number between 0 and 1, which means the ratio of validation frames, or an integer number which means the exact number of validation frames')
    parser.add_argument('--ignore_module_load', default=[], type=str, nargs='+')
    parser.add_argument('--layers', default=5, type=int)
    parser.add_argument('--random', default=0, type=float)
    parser.add_argument('--adaptive_quant', default=0, type=int)
    parser.add_argument('--bar', default=0, type=int, help='set to 1 to use STE for quantization, set to 0 to add noise in replacement of quantization')
    parser.add_argument('--save_every', default=10, type=int, help='save frequency (unit: epoch)')
    parser.add_argument('--octree_end', default=2, type=int, help='any nodes in layers lower than octree_end will be surfel nodes')
    parser.add_argument('--surfel_start', default=3, type=int, help='any nodes in layers higher than surfel_start will be octree nodes')
    parser.add_argument('--selection_mode', default='fixed', type=str, help='selection mode for decision module, can be fixed or gumbel')
    parser.add_argument('--from_step', default=0, type=int, help='from which step to start training, this relates to the temperature of gumbel softmax')
    parser.add_argument('--scheduler_step_size', default=10, type=int, help='step size for learning rate decay')
    return parser.parse_args()
args = parse_args()
import os
import sys

if not args.exp_name:
    args.exp_name = args.model + '_I' + str(int(args.lamb))
    if args.augment:
        args.exp_name += '_a'

import importlib
import logging
import random
import shutil
import glob
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import StaticDataset, static_collate_pointcloud_fn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, os.pardir))
import MinkowskiEngine as ME


SUPPORTED_DATA_EXTENSIONS = ("h5", "ply", "bin", "npy")


def log_string(message):
    """Write the same message to the experiment log and standard output."""
    logger.info(message)
    print(message)


def checkpoint_state(epoch, loss, model, optimizer):
    """Build the checkpoint payload used by every save location."""
    return {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }


if __name__ == '__main__':

    # Experiment directories
    if args.exp_name is None:
        args.exp_name = ''.join(random.sample('abcdefghijklmnopqrstuvwxyz', 10))
    log_root_dir = Path(args.root_dir)
    log_root_dir.mkdir(exist_ok=True)
    model_dir = log_root_dir.joinpath(args.model)
    model_dir.mkdir(exist_ok=True)
    experiment_dir = model_dir.joinpath(args.exp_name)
    experiment_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('log')
    checkpoint_dir = experiment_dir.joinpath('checkpoint')
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    # Logger
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # Dataset
    all_filedirs = []
    for dataset_dir in args.dataset_dir:
        filedirs = sorted(
            glob.glob(os.path.join(dataset_dir, '**', '*.*'), recursive=True)
        )
        all_filedirs = all_filedirs + [
            filedir
            for filedir in filedirs
            if filedir.endswith(SUPPORTED_DATA_EXTENSIONS)
        ]
    val_num = int(len(all_filedirs) * args.val_num)
    val_filedirs = all_filedirs[::int(len(all_filedirs) // val_num) + 1]
    print('val filedirs\n', val_filedirs)
    train_filedirs = [item for item in all_filedirs if item not in val_filedirs]
    print('total', len(train_filedirs), 'frames in training set')

    val_dataset = StaticDataset(seqs_list=val_filedirs, attribute=False, yuv=1, crop=-1, type='val')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=static_collate_pointcloud_fn,
    )


    # Model
    MODEL = importlib.import_module(args.model)
    shutil.copy('./%s.py' % args.model, str(experiment_dir))
    shutil.copy('./trainer.py', str(experiment_dir))
    shutil.copy('./dataset.py', str(experiment_dir))
    shutil.copy('./resnet.py', str(experiment_dir))
    shutil.copy('./models.py', str(experiment_dir))
    model = MODEL.get_model(
        channels=args.channels,
        hyper=args.hyper,
        bar=args.bar,
        octree_end=args.octree_end,
        surfel_start=args.surfel_start,
        adaptive_quant=args.adaptive_quant,
        selection_mode=args.selection_mode,
    )

    # Load existing model weights under the experiment directory.
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoint/model_last.pth', map_location='cuda:0')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Detected existed model')
        exist = True
    except:
        log_string('No existing model')
        start_epoch = 0
        exist = False

    if len(args.pretrained) != 0:
        checkpoint = torch.load(args.pretrained)
        start_epoch = 0
        import collections

        old_paras = model.state_dict()
        new_state_dict = collections.OrderedDict()
        checkpoint_items = (
            checkpoint['model_state_dict'].items()
            if 'model_state_dict' in checkpoint
            else checkpoint.items()
        )
        for k, v in checkpoint_items:
            k1 = k.replace('module.', '')
            if k1 not in old_paras:
                print(k1, 'not exist')
                continue

            if old_paras[k1].size() != v.size():
                print(k1, old_paras[k1].size(), v.size(), 'not matched')
                continue
            new_state_dict[k1] = v

        old_paras.update(new_state_dict)
        model.load_state_dict(old_paras)
        log_string('Finetuning')

    model = model.cuda()

    # Optimizer
    params_lr_list = []
    for module_name in model._modules.keys():
        if args.frozen_modules and module_name in args.frozen_modules:
            for parameter in model._modules[module_name].parameters():
                parameter.requires_grad = False
            logger.info('frozen: ' + module_name + '\tlr:0')
        else:
            lr = args.learning_rate
            params_lr_list.append({"params": model._modules[module_name].parameters(), 'lr': lr})
            logger.info('optimize: ' + module_name + '\tlr:' + str(lr))

    optimizer = torch.optim.Adam(
        params_lr_list,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    if exist:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log_string('use exist optimizer')
        best_loss_test = 99999999
    else:
        best_loss_test = 99999999
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.7)
    for _ in range(start_epoch % 10):  # Preserve the original resume behavior.
        scheduler.step()
    train_dataset = StaticDataset(seqs_list=train_filedirs, attribute=True, yuv=1, crop=800000)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.actual_batch_size,
        shuffle=True,
        collate_fn=static_collate_pointcloud_fn,
        num_workers=8,
    )

    print(optimizer)

    # Training
    log_string('start training')
    step = 0
    total_step = args.from_step
    for epoch in range(start_epoch, args.epoch):
        log_string('\nEpoch: ' + str(epoch + 1))
        total_bpp = 0
        total_cls = 0
        total_loss = 0
        counter = 0
        model.train()
        device = torch.device('cuda')

        for batch_id, data in tqdm(
            enumerate(train_dataloader),
            total=min(len(train_dataloader), args.max_iter),
            smoothing=0.9,
        ):
            lamb = args.lamb
            xyz, color = data
            xyz, color = xyz.to(torch.float32), color.to(torch.float32)
            color = torch.ones_like(color[:, :1])
            f1 = ME.SparseTensor(features=color, coordinates=xyz, device=device)
            f2 = ME.SparseTensor(features=color, coordinates=xyz, device=device)

            output = model(f1, f2, device, epoch, lamb=lamb, keep_adaptive=True, step=total_step)
            bce = output['bce']
            bpp = output['bpp']
            distortion = bce
            loss = distortion * lamb + bpp

            step = (step + 1) % (args.batch_size // args.actual_batch_size)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            if step == 0:
                optimizer.step()
                optimizer.zero_grad()
                total_step += 1

            total_bpp += bpp.item()
            total_cls += distortion.item()
            total_loss += loss.item()
            print('lamb bpp cls', lamb, bpp.item(), distortion.item())
            counter += 1
            if batch_id > args.max_iter:
                break

            torch.cuda.empty_cache()
            break


        avg_bpp = total_bpp / counter
        avg_cls = total_cls / counter
        avg_loss = total_loss / counter

        log_string('\naverage bpp: ' + str(avg_bpp))
        log_string('\naverage cls: ' + str(avg_cls))
        log_string('\naverage loss: ' + str(avg_loss))
        scheduler.step()
        savepath = str(checkpoint_dir) + '/model_last.pth'
        state = checkpoint_state(epoch, avg_bpp, model, optimizer)
        torch.save(state, savepath)
        if epoch % args.save_every == 0:
            savepath = str(checkpoint_dir) + '/' + str(epoch) + '.pth'
            state = checkpoint_state(epoch, avg_bpp, model, optimizer)
            torch.save(state, savepath)
        log_string('evaluating')
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        test_total_bpp = 0
        test_total_cls = 0
        test_total_loss = 0
        test_counter = 0
        with torch.no_grad():
            lambTest = args.lamb
            for batch_id, data in tqdm(
                enumerate(val_dataloader, 0),
                total=len(val_dataloader),
                smoothing=0.9,
            ):
                xyz, point = data
                xyz, point = xyz.to(torch.float32), point.to(torch.float32)
                f2 = ME.SparseTensor(features=point, coordinates=xyz, device=device)
                output = model(f2, f2, device, epoch, lamb=lambTest, keep_adaptive=True)
                bce = output['bce']
                bpp = output['bpp']
                distortion = bce
                loss = distortion * lambTest + bpp

                test_total_bpp += bpp.item()
                test_total_loss += loss.item()
                test_total_cls += distortion.item()
                test_counter += 1

                torch.cuda.empty_cache()

        test_avg_bpp = test_total_bpp / test_counter
        test_avg_cls = test_total_cls / test_counter
        test_avg_loss = test_total_loss / test_counter
        if test_avg_loss < best_loss_test:
            logger.info('Save model...')
            best_loss_test = test_avg_loss
            savepath = str(checkpoint_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = checkpoint_state(epoch, test_avg_bpp, model, optimizer)
            torch.save(state, savepath)
            torch.save(model.state_dict(), str(checkpoint_dir) + '/best_model_weights_only.pth')
        log_string('\ntest average bpp: ' + str(test_avg_bpp))
        log_string('\ntest average cls: ' + str(test_avg_cls))
        log_string('\ntest average loss: ' + str(test_avg_loss))
