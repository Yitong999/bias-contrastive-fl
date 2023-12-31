import argparse
import datetime
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from debias.datasets.biased_mnist_fl import get_color_mnist
from debias.losses.diloss import DILoss
from debias.networks.simple_conv_di import DISimpleConvNet
from debias.utils.logging import set_logging
from debias.utils.utils import (AverageMeter, MultiDimAverageMeter, accuracy,
                                save_model, set_seed, pretty_dict)


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test', )
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--print_freq', type=int, default=300,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--epochs', type=int, default=80,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--corr', type=float, default=0.999)

    parser.add_argument('--bs', type=int, default=128, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-3)

    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    return opt


def set_model():
    model = DISimpleConvNet().cuda()
    criterion = DILoss(num_classes=10, num_biases=10)

    return model, criterion


def train(train_loader, model, criterion, optimizer):
    model.train()
    avg_loss = AverageMeter()

    train_iter = iter(train_loader)
    for idx, (images, labels, biases, _) in enumerate(train_iter):
        bsz = labels.shape[0]
        labels, biases = labels.cuda(), biases.cuda()

        images = images.cuda()
        logits = model(images)

        loss = criterion(logits, labels, biases)

        avg_loss.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_loss.avg


def validate(val_loader, model):
    model.eval()

    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(10, 10))
    num_biases = 10
    num_classes = 10

    with torch.no_grad():
        for idx, (images, labels, biases, _) in enumerate(val_loader):
            images, labels, biases = images.cuda(), labels.cuda(), biases.cuda()
            bsz = labels.shape[0]

            logits = model(images)
            logits = sum([F.log_softmax(logits[:, i * num_classes: (i + 1) * num_classes], dim=1) for i in
                          range(num_biases)])
            preds = logits.data.max(1, keepdim=True)[1].squeeze(1)

            acc1, = accuracy(logits, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            corrects = (preds == labels).long()
            attrwise_acc_meter.add(corrects.cpu(), torch.stack([labels.cpu(), biases.cpu()], dim=1))

    return top1.avg, attrwise_acc_meter.get_unbiased_acc()


def main():
    opt = parse_option()

    exp_name = f'di-color_mnist_corr{opt.corr}-{opt.exp_name}-lr{opt.lr}-bs{opt.bs}-seed{opt.seed}'
    opt.exp_name = exp_name

    output_dir = f'exp_results/{exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, 'INFO', str(save_path))
    logging.info(f'Set seed: {opt.seed}')
    set_seed(opt.seed)
    logging.info(f'save_path: {save_path}')

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)
    
    root = './data/biased_mnist'
    train_loader = get_color_mnist(
        root,
        batch_size=opt.bs,
        data_label_correlation=opt.corr,
        n_confusing_labels=9,
        split='train',
        seed=opt.seed,
        aug=False, )
    logging.info(
        f'confusion_matrix - \n original: {train_loader.dataset.confusion_matrix_org}, \n normalized: {train_loader.dataset.confusion_matrix}')

    val_loaders = {}
    val_loaders['valid'] = get_color_mnist(
        root,
        batch_size=256,
        data_label_correlation=0.1,
        n_confusing_labels=9,
        split='train_val',
        seed=opt.seed,
        aug=False,
    )
    val_loaders['test'] = get_color_mnist(
        root,
        batch_size=256,
        data_label_correlation=0.1,
        n_confusing_labels=9,
        split='valid',
        seed=opt.seed,
        aug=False)

    model, criterion = set_model()

    decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    logging.info(f"decay_epochs: {decay_epochs}")

    (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)

    best_accs = {'valid': 0, 'test': 0}
    best_epochs = {'valid': 0, 'test': 0}
    best_stats = {}
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        logging.info(f'[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}')
        loss = train(train_loader, model, criterion, optimizer)
        logging.info(f'[{epoch} / {opt.epochs}] Loss: {loss}')

        scheduler.step()

        stats = pretty_dict(epoch=epoch)
        for key, val_loader in val_loaders.items():
            _, acc_unbiased = validate(val_loader, model)
            stats[f'{key}/acc_unbiased'] = acc_unbiased.item() * 100

        logging.info(f'[{epoch} / {opt.epochs}] {stats}')
        for tag in best_accs.keys():
            if stats[f'{tag}/acc_unbiased'] > best_accs[tag]:
                best_accs[tag] = stats[f'{tag}/acc_unbiased']
                best_epochs[tag] = epoch
                best_stats[tag] = pretty_dict(**{f'best_{tag}_{k}': v for k, v in stats.items()})

                save_file = save_path / 'checkpoints' / f'best_{tag}.pth'
                save_model(model, optimizer, opt, epoch, save_file)
            logging.info(
                f'[{epoch} / {opt.epochs}] best {tag} accuracy: {best_accs[tag]:.3f} at epoch {best_epochs[tag]} \n best_stats: {best_stats[tag]}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Total training time: {total_time_str}')

    save_file = save_path / 'checkpoints' / f'last.pth'
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
