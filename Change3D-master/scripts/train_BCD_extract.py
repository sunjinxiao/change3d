# Copyright (c) Duowang Zhu.
# All rights reserved.

import os
import sys
import time
import numpy as np
from os.path import join as osp
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# Insert current path for local module imports
sys.path.insert(0, '.')

import data.dataset as RSDataset
import torch.nn as nn
import data.transforms as RSTransforms
from utils.metric_tool import ConfuseMatrixMeter

from model.trainer import Trainer
from model.utils import (
    adjust_learning_rate,
    BCEDiceLoss,
    load_checkpoint,
    setup_logger
)


class GateBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(GateBlock, self).__init__()
        num_gates = in_channels
        # 修正：防止 3通道图像整除 16 变成 0，限制最小通道数为 1
        mid_channels = max(1, in_channels // reduction)

        # 平均池化分支
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1_avg = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.norm1_avg = nn.LayerNorm((mid_channels, 1, 1))
        self.relu = nn.ReLU(inplace=True)

        # 最大池化分支
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1_max = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.norm1_max = nn.LayerNorm((mid_channels, 1, 1))

        # 两路汇合
        self.fc2 = nn.Conv2d(mid_channels, num_gates, kernel_size=1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg = self.relu(self.norm1_avg(self.fc1_avg(self.global_avgpool(x))))
        mx = self.relu(self.norm1_max(self.fc1_max(self.global_maxpool(x))))
        x = self.gate_activation(self.fc2(avg + mx))
        return input * x

class StyleStrip(nn.Module):
    def __init__(self, in_channels):
        super(StyleStrip, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(in_channels, affine=True)
        self.style_strip = GateBlock(in_channels, reduction=16)

    def forward(self, x):
        x_IN_1 = self.instance_norm(x)     # 获取剔除风格后的内容
        x_style_1 = x - x_IN_1             # 获取被剥离的风格残差
        x_style_1_useful = self.style_strip(x_style_1) # 筛选有用的风格
        x_1 = x_IN_1 + x_style_1_useful    # 加回内容中
        return x_1

def create_data_loaders(args, train_transform, val_transform):
    """
    Creates data loaders for training, validation, and testing.

    Args:
        args: Command line arguments.
        train_transform: Transform pipeline for training data.
        val_transform: Transform pipeline for validation and testing data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, max_batches).
    """
    # Training data
    train_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="train",
        transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Validation data
    val_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="val",
        transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Test data
    test_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="test",
        transform=val_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    max_batches = len(train_loader)
    print(f"For each epoch, we have {max_batches} batches.")

    return train_loader, val_loader, test_loader, max_batches


@torch.no_grad()
def val(args, val_loader, model,style_strip, epoch):
    """
    Validates the model on the validation set.

    Args:
        args: Command line arguments.
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): The model to validate.
        epoch (int): Current epoch index.

    Returns:
        tuple: (average_loss, scores).
    """
    model.eval()
    style_strip.eval()  # 开启评估模式sun
    eval_meter = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    total_batches = len(val_loader)

    print(f"Validation on {total_batches} batches")

    for iter_idx, batched_inputs in enumerate(val_loader):
        img, target = batched_inputs

        # Simplified data preparation
        pre_img = img[:, 0:3].cuda().float()
        post_img = img[:, 3:6].cuda().float()
        target = target.cuda().float()
        # ================= 新增：验证集也必须经过相同的风格处理 sun=================
        pre_img = style_strip(pre_img)
        post_img = style_strip(post_img)
        # =====================================================================
        start_time = time.time()

        # Forward pass
        output = model.update_bcd(pre_img, post_img)
        loss = BCEDiceLoss(output, target)

        # Binarize predictions
        pred = torch.where(
            output > 0.5,
            torch.ones_like(output),
            torch.zeros_like(output)
        ).long()

        time_taken = time.time() - start_time
        epoch_loss.append(loss.data.item())

        # Update evaluation metrics
        f1 = eval_meter.update_cm(
            pr=pred.cpu().numpy(),
            gt=target.cpu().numpy()
        )

        if iter_idx % 5 == 0:
            print(
                f"\r[{iter_idx}/{total_batches}] "
                f"F1: {f1:.3f} loss: {loss.data.item():.3f} "
                f"time: {time_taken:.3f}",
                end=''
            )

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = eval_meter.get_scores()

    return average_epoch_loss_val, scores


def train(args, train_loader, model, style_strip,optimizer, epoch, max_batches,
          cur_iter=0, lr_factor=1.):
    """
    Trains the model for one epoch.

    Args:
        args: Command line arguments.
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): Model to train.
        optimizer: Optimizer instance.
        epoch (int): Current epoch index.
        max_batches (int): Number of batches per epoch.
        cur_iter (int): Current iteration count.
        lr_factor (float): Learning rate adjustment factor.

    Returns:
        tuple: (average_loss, scores, current_lr).
    """
    model.train()
    style_strip.train()  # 确保风格模块也开启训练模式sun
    eval_meter = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    for iter_idx, batched_inputs in enumerate(train_loader):
        img, target = batched_inputs

        # Simplified data preparation
        pre_img = img[:, 0:3].cuda().float()
        post_img = img[:, 3:6].cuda().float()
        target = target.cuda().float()
        # ================= 新增：图像送入主网络前进行风格门控 sun=================
        pre_img = style_strip(pre_img)
        post_img = style_strip(post_img)
        # ====================================================================
        start_time = time.time()

        # Adjust learning rate
        lr = adjust_learning_rate(
            args,
            optimizer,
            epoch,
            iter_idx + cur_iter,
            max_batches,
            lr_factor=lr_factor
        )

        # Forward pass
        output = model.update_bcd(pre_img, post_img)
        loss = BCEDiceLoss(output, target)

        # Binarize predictions
        pred = torch.where(
            output > 0.5,
            torch.ones_like(output),
            torch.zeros_like(output)
        ).long()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter_idx - cur_iter) * time_taken / 3600

        # Update metrics
        with torch.no_grad():
            f1 = eval_meter.update_cm(
                pr=pred.cpu().numpy(),
                gt=target.cpu().numpy()
            )

        if (iter_idx + 1) % 5 == 0:
            print(
                f"[epoch {epoch}] [iter {iter_idx + 1}/{len(train_loader)} {res_time:.2f}h] "
                f"[lr {optimizer.param_groups[0]['lr']:.6f}] "
                f"[bn_loss {loss.data.item():.4f}] "
            )

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    scores = eval_meter.get_scores()

    return average_epoch_loss_train, scores, lr


def trainValidate(args):
    """
    Main training and validation routine.

    Args:
        args: Command line arguments.
    """
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Enable CUDA optimizations and fix random seed
    torch.backends.cudnn.benchmark = True
    cudnn.benchmark = True
    torch.manual_seed(seed=16)
    torch.cuda.manual_seed(seed=16)

    # Initialize model
    model = Trainer(args).cuda().float()

    # ================= sun新增：初始化 StyleStrip 并移至 GPU =================
    style_strip = StyleStrip(in_channels=3).cuda().float()
    # ====================================================================
#sun
    # Create experiment save directory
    save_path = osp(
        args.save_dir,
        "LEVIR-256-extract"
    )
    os.makedirs(save_path, exist_ok=True)

    # Data transformations
    train_transform, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)

    # Data loaders
    train_loader, val_loader, test_loader, max_batches = create_data_loaders(
        args, train_transform, val_transform
    )

    # Compute maximum epochs
    args.max_epochs = int(np.ceil(args.max_steps / max_batches))

    # Load checkpoint if needed
    start_epoch, cur_iter = load_checkpoint(args, model, save_path, max_batches)

    # Set up logger
    logger = setup_logger(args, save_path)

    # Create optimizer
    optimizer = torch.optim.Adam(
        #sun
        [
            {'params': model.parameters()},
            {'params': style_strip.parameters()}
        ],
        #sun
        args.lr,
        (0.9, 0.99),
        eps=1e-08,
        weight_decay=1e-4
    )

    # Track best F1 score
    max_F1_val = 0

    # Main training loop
    for epoch in range(start_epoch, args.max_epochs):
        torch.cuda.empty_cache()

        # Train one epoch
        loss_train, score_tr, lr = train(
            args,
            train_loader,
            model,
            style_strip,
            optimizer,
            epoch,
            max_batches,
            cur_iter
        )
        cur_iter += len(train_loader)

        # Skip validation for the first epoch
        if epoch == 0:
            continue

        # Validation (using test set as validation)
        torch.cuda.empty_cache()
        loss_val, score_val = val(args, test_loader, model, style_strip,epoch)

        # Log validation results
        logger.write(
            "\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
                epoch,
                score_val['Kappa'],
                score_val['IoU'],
                score_val['F1'],
                score_val['recall'],
                score_val['precision']
            )
        )
        logger.flush()

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_val': loss_val,
            'F_train': score_tr['F1'],
            'F_val': score_val['F1'],
            'lr': lr
        }, osp(save_path, 'checkpoint.pth.tar'))

        # Save the best model
        model_file_name = osp(save_path, 'best_model.pth')
        style_file_name = osp(save_path, 'best_style_strip.pth')  # 新增保存路径sun
        if epoch % 1 == 0 and max_F1_val <= score_val['F1']:
            max_F1_val = score_val['F1']
            torch.save(model.state_dict(), model_file_name)
            torch.save(style_strip.state_dict(), style_file_name)  # 保存 style_strip 的权重sun

        # Print summary
        print(f"\nEpoch {epoch}: Details")
        print(
            f"\nEpoch No. {epoch}:\tTrain Loss = {loss_train:.4f}\t"
            f"Val Loss = {loss_val:.4f}\tF1(tr) = {score_tr['F1']:.4f}\t"
            f"F1(val) = {score_val['F1']:.4f}"
        )

    # Test with the best model
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)
    style_strip.load_state_dict(torch.load(style_file_name))  # 加载 best_style_strip  sun

    loss_test, score_test = val(args, test_loader, model,style_strip, 0)
    print(
        f"\nTest:\t Kappa (te) = {score_test['Kappa']:.4f}\t "
        f"IoU (te) = {score_test['IoU']:.4f}\t"
        f"F1 (te) = {score_test['F1']:.4f}\t "
        f"R (te) = {score_test['recall']:.4f}\t"
        f"P (te) = {score_test['precision']:.4f}"
    )

    logger.write(
        "\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
            'Test',
            score_test['Kappa'],
            score_test['IoU'],
            score_test['F1'],
            score_test['recall'],
            score_test['precision']
        )
    )
    logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset',
        default="LEVIR-CD",
        help='Dataset selection | LEVIR-CD | WHU-CD | CLCD'
    )
    parser.add_argument(
        '--file_root',
        default="path/to/LEVIR-CD",
        help='path to the dataset directory'
    )
    parser.add_argument(
        '--in_height',
        type=int,
        default=256,
        help='Height of RGB image'
    )
    parser.add_argument(
        '--in_width',
        type=int,
        default=256,
        help='Width of RGB image'
    )
    parser.add_argument(
        '--num_perception_frame',
        type=int,
        default=1,
        help='Number of perception frames'
    )
    parser.add_argument(
        '--num_class',
        type=int,
        default=1,
        help='Number of classes'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=80000,
        help='Max number of iterations'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of parallel threads'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-4,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--lr_mode',
        default='poly',
        help='Learning rate policy: step or poly'
    )
    parser.add_argument(
        '--step_loss',
        type=int,
        default=100,
        help='Decrease learning rate after how many epochs'
    )
    parser.add_argument(
        '--pretrained',
        default='model/X3D_L.pyth',
        type=str,
        help='Path to pretrained weight'
    )
    parser.add_argument(
        '--save_dir',
        default='./exp',
        help='Directory to save the experiment results'
    )
    parser.add_argument(
        '--resume',
        default=None,
        help='Checkpoint to resume training'
    )
    parser.add_argument(
        '--log_file',
        default='train_val_log.txt',
        help='File that stores the training and validation logs'
    )
    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int,
        help='GPU ID number'
    )

    args = parser.parse_args()
    trainValidate(args)