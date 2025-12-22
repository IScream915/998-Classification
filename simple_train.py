import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
import logging

from model.repghost import repghostnet_0_5x, repghostnet_0_8x, repghostnet_1_0x

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='Simple Training for RepGhost with Custom Dataset')

    # 数据集参数
    parser.add_argument('--train_dir', type=str, default='dataset/train',
                        help='训练数据目录')
    parser.add_argument('--val_dir', type=str, default='dataset/val',
                        help='验证数据目录')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='类别数量')

    # 模型参数
    parser.add_argument('--model_size', type=str, default='0_5x',
                        choices=['0_5x', '0_8x', '1_0x'],
                        help='模型大小')
    parser.add_argument('--pretrained', action='store_true',
                        help='是否使用预训练权重')

    # 训练参数 - 为M3芯片优化
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小 (M3芯片建议使用较小的批次)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--img_size', type=int, default=224,
                        help='输入图像大小')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='outputs/train',
                        help='输出目录')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='模型保存间隔（轮数）')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='日志打印间隔（批次数）')

    # 其他参数 - M3优化
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载器工作进程数 (M3建议使用0避免多进程问题)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='训练设备 (M3芯片使用cpu或mps)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    return parser.parse_args()


def set_device(device_arg):
    """设置训练设备，支持M3芯片的MPS"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f'使用GPU训练: {torch.cuda.get_device_name()}')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info('使用Apple Silicon MPS加速训练 (M3芯片)')
        else:
            device = torch.device('cpu')
            logger.info('使用CPU训练')
    else:
        device = torch.device(device_arg)
        if device.type == 'mps':
            logger.info('使用Apple Silicon MPS加速训练 (M3芯片)')
        elif device.type == 'cuda':
            logger.info(f'使用GPU训练: {torch.cuda.get_device_name()}')
        else:
            logger.info('使用CPU训练')

    # 设置PyTorch优化选项
    if device.type == 'cpu':
        # CPU优化设置
        torch.set_num_threads(4)  # M3芯片有8个性能核心，使用4个线程避免过载
        logger.info('CPU优化: 设置线程数为4')

    return device


def get_model(model_size, num_classes, pretrained=False):
    """获取模型"""
    if model_size == '0_5x':
        model = repghostnet_0_5x(num_classes=num_classes)
    elif model_size == '0_8x':
        model = repghostnet_0_8x(num_classes=num_classes)
    elif model_size == '1_0x':
        model = repghostnet_1_0x(num_classes=num_classes)
    else:
        raise ValueError(f'不支持的模型大小: {model_size}')

    if pretrained:
        logger.info('注意: 当前没有实现预训练权重加载，将使用随机初始化')

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'模型参数总数: {total_params:,}')
    logger.info(f'可训练参数: {trainable_params:,}')

    return model


def get_data_loaders(args):
    """获取数据加载器"""
    # 训练数据变换
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证数据变换
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = ImageFolder(root=args.train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=args.val_dir, transform=val_transform)

    logger.info(f'训练集样本数: {len(train_dataset)}')
    logger.info(f'验证集样本数: {len(val_dataset)}')
    logger.info(f'类别: {train_dataset.classes}')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, train_dataset.classes


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.log_interval == 0:
            logger.info(f'Epoch [{epoch}/{args.epochs}] '
                       f'Batch [{batch_idx}/{len(train_loader)}] '
                       f'Loss: {loss.item():.4f} '
                       f'Acc: {100.*correct/total:.2f}%')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, acc, args, is_best=False):
    """保存检查点"""
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'acc': acc,
        'args': args
    }

    # 保存最新检查点
    torch.save(checkpoint, os.path.join(args.output_dir, 'latest_checkpoint.pth'))

    # 定期保存检查点
    if (epoch + 1) % args.save_interval == 0:
        torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))

    # 保存最佳模型
    if is_best:
        torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))


def setup_logging(output_dir):
    """设置日志文件保存"""
    os.makedirs(output_dir, exist_ok=True)

    # 清除默认的handler
    logger.handlers.clear()

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件handler
    log_file = os.path.join(output_dir, f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f'日志文件保存路径: {log_file}')
    return log_file


def main():
    args = get_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    log_file = setup_logging(args.output_dir)

    # 保存训练参数
    import json
    args_file = os.path.join(args.output_dir, 'training_args.json')
    with open(args_file, 'w', encoding='utf-8') as f:
        # 将argparse.Namespace转换为字典，处理不可序列化的对象
        args_dict = vars(args)
        json.dump(args_dict, f, indent=2, ensure_ascii=False)
    logger.info(f'训练参数保存到: {args_file}')

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 设置设备
    device = set_device(args.device)

    # 获取数据加载器
    logger.info('加载数据集...')
    train_loader, val_loader, class_names = get_data_loaders(args)

    # 保存类别信息
    class_file = os.path.join(args.output_dir, 'class_names.txt')
    with open(class_file, 'w', encoding='utf-8') as f:
        for class_name in class_names:
            f.write(f'{class_name}\n')
    logger.info(f'类别信息保存到: {class_file}')

    # 获取模型
    logger.info('初始化模型...')
    model = get_model(args.model_size, args.num_classes, args.pretrained)
    model = model.to(device)

    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 训练历史记录
    history_file = os.path.join(args.output_dir, 'training_history.csv')
    with open(history_file, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc,lr,best_acc\n')

    # 训练循环
    logger.info('开始训练...')
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            logger.info(f'新的最佳验证准确率: {best_acc:.2f}%')

        save_checkpoint(model, optimizer, epoch, val_acc, args, is_best)

        # 记录训练历史
        epoch_time = time.time() - epoch_start_time
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(f'{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f},{current_lr:.6f},{best_acc:.2f}\n')

        # 记录每个epoch的结果
        logger.info(f'Epoch [{epoch+1}/{args.epochs}] '
                   f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% '
                   f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% '
                   f'LR: {current_lr:.6f} Time: {epoch_time:.2f}s')

    total_time = time.time() - start_time
    logger.info(f'训练完成! 最佳验证准确率: {best_acc:.2f}%')
    logger.info(f'总训练时间: {total_time/3600:.2f} 小时 ({total_time:.1f} 秒)')

    # 保存最终训练摘要
    summary_file = os.path.join(args.output_dir, 'training_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f'训练完成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'总训练时间: {total_time/3600:.2f} 小时\n')
        f.write(f'训练轮数: {args.epochs}\n')
        f.write(f'最佳验证准确率: {best_acc:.2f}%\n')
        f.write(f'模型大小: {args.model_size}\n')
        f.write(f'批次大小: {args.batch_size}\n')
        f.write(f'学习率: {args.lr}\n')
        f.write(f'设备: {device}\n')
        f.write(f'训练集样本数: {len(train_loader.dataset)}\n')
        f.write(f'验证集样本数: {len(val_loader.dataset)}\n')
        f.write(f'类别数: {args.num_classes}\n')

    logger.info(f'训练摘要保存到: {summary_file}')


if __name__ == '__main__':
    main()