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
import data.transforms as RSTransforms
from utils.metric_tool import ConfuseMatrixMeter

from model_loss.trainer import Trainer
from model_loss.utils import (
    adjust_learning_rate,
    BCEDiceLoss,
    load_checkpoint,
    setup_logger
)

# 在文件头部添加
import torch.nn.functional as F


def feature_contrastive_loss(preds, features, targets, tau=0.5):
    """
    计算 L_pseudo 和 L_true
    preds: [B, 1, H, W] - 网络输出的概率图
    features: [B, C, H, W] - 对应的最终高维特征 (C维度通常为24/64/256等)
    targets: [B, 1, H, W] 或 [B, H, W] - 真实标签
    """
    B, C, H, W = features.shape

    # 维度对齐保护
    if len(targets.shape) == 3:
        targets = targets.unsqueeze(1)

    features_flat = features.view(B, C, -1)  # [B, C, N]
    targets_flat = targets.view(B, 1, -1)  # [B, 1, N]
    preds_flat = preds.view(B, 1, -1)  # [B, 1, N]

    # === 1. 计算当前Batch的特征原型 (Prototypes) ===
    bg_mask = 1.0 - targets_flat
    fg_mask = targets_flat

    # 利用真实标签提取出纯净的背景和变化原型 [B, C]
    proto_bg = (features_flat * bg_mask).sum(dim=2) / (bg_mask.sum(dim=2) + 1e-8)
    proto_fg = (features_flat * fg_mask).sum(dim=2) / (fg_mask.sum(dim=2) + 1e-8)

    # 关键：必须 detach 原型！我们不希望被拉扯的错误特征反过来污染原型中心
    proto_bg = F.normalize(proto_bg.detach(), p=2, dim=1, eps=1e-8)
    proto_fg = F.normalize(proto_fg.detach(), p=2, dim=1, eps=1e-8)

    # === 2. 找到模型判断错的区域 ===
    pred_binary = (preds_flat > tau).float()
    m_pseudo = pred_binary * (1.0 - targets_flat)  # P>tau 但 GT=0 (伪变化)
    m_true = pred_binary * targets_flat  # P>tau 且 GT=1 (真实变化)

    # 特征归一化用于计算余弦相似度
    features_norm = F.normalize(features_flat, p=2, dim=1, eps=1e-8)  # [B, C, N]

    # === 3. 计算 L_pseudo (推向 bg) ===
    # 计算特征与背景原型的余弦相似度 [B, 1, N]
    sim_pseudo_bg = (features_norm * proto_bg.unsqueeze(2)).sum(dim=1, keepdim=True)
    # Loss = (1 - sim)，相似度越低(差异越大)，Loss越大
    loss_pseudo = ((1.0 - sim_pseudo_bg) * m_pseudo).sum() / (m_pseudo.sum() + 1e-8)

    # === 4. 计算 L_true (推向 fg) ===
    sim_true_fg = (features_norm * proto_fg.unsqueeze(2)).sum(dim=1, keepdim=True)
    loss_true = ((1.0 - sim_true_fg) * m_true).sum() / (m_true.sum() + 1e-8)

    return loss_pseudo, loss_true

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
def val(args, val_loader, model, epoch):
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

        start_time = time.time()

        # # Forward pass
        # output = model.update_bcd(pre_img, post_img)
        # loss = BCEDiceLoss(output, target)

        # 修改为：sun
        output, features = model.update_bcd(pre_img, post_img, return_features=True)
        loss_main = BCEDiceLoss(output, target)

        # 验证集单纯只看主干 loss 即可，或者你也把伪标签 loss 加上
        loss = loss_main
        #sun

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


def train(args, train_loader, model, optimizer, epoch, max_batches, 
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
    eval_meter = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    for iter_idx, batched_inputs in enumerate(train_loader):
        img, target = batched_inputs

        # Simplified data preparation
        pre_img = img[:, 0:3].cuda().float()
        post_img = img[:, 3:6].cuda().float()
        target = target.cuda().float()

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
        # output = model.update_bcd(pre_img, post_img)
        # loss = BCEDiceLoss(output, target)
        # 修改为：sun
        # 设置超参数 (建议在主函数通过 parser 传入，这里默认0.1)
        lambda_1 = 0.1
        lambda_2 = 0.1

        output, features = model.update_bcd(pre_img, post_img, return_features=True)

        # 1. 主干 Loss (BCE + Dice)
        loss_main = BCEDiceLoss(output, target)

        # 2. 特征空间监督 Loss
        loss_pseudo, loss_true = feature_contrastive_loss(output, features, target)

        # 3. 总 Loss
        loss = loss_main + lambda_1 * loss_pseudo + lambda_2 * loss_true
        # sun

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

    # Create experiment save directory
    save_path = osp(
        args.save_dir,
        "LEVIR-loss"
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
        model.parameters(),
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
        loss_val, score_val = val(args, test_loader, model, epoch)
        
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
        if epoch % 1 == 0 and max_F1_val <= score_val['F1']:
            max_F1_val = score_val['F1']
            torch.save(model.state_dict(), model_file_name)

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

    loss_test, score_test = val(args, test_loader, model, 0)
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