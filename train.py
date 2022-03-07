#!/usr/bin/env python3
import argparse
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim

import dataset
import models
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--data', default='inat21_mini', type=str, help='inat21_mini|inat21_full')
    parser.add_argument('--data_dir', default='datasets/inat21', type=str)
    parser.add_argument('--save_dir', default='./logs', type=str)
    parser.add_argument('--model_file', default='sk2res2net_dynamic_mlp', type=str, help='model file name')
    parser.add_argument('--model_name', default='sk2res2net101', type=str, help='model type in detail')
    parser.add_argument('--fold', default=1, type=int, help='training fold')
    parser.add_argument('--random_seed', default=37, type=int)

    # train
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--warmup', default=2, type=int)
    parser.add_argument('--start_lr', default=0.04, type=float)
    parser.add_argument('--stop_epoch', default=90, type=int)
    parser.add_argument('--num_workers', default=32, type=int)

    # data
    parser.add_argument('--tencrop', action='store_true', default=False)
    parser.add_argument('--image_only', action='store_true', default=False)
    parser.add_argument('--metadata', default='geo_temporal', type=str, help='geo_temporal|geo|temporal')

    # model
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')

    # dynamic MLP
    parser.add_argument('--mlp_type', default='c', type=str, help='dynamic mlp versions: a|b|c')
    parser.add_argument('--mlp_d', default=256, type=int)
    parser.add_argument('--mlp_h', default=64, type=int)
    parser.add_argument('--mlp_n', default=2, type=int)

    args = parser.parse_args()
    args.mlp_cin = 0
    if 'geo' in args.metadata:
        args.mlp_cin += 4
    if 'temporal' in args.metadata:
        args.mlp_cin += 2

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    args.nprocs = torch.cuda.device_count()

    # get logger
    creat_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    args.path_log = os.path.join(args.save_dir, f'{args.data}', f'{args.name}')
    os.makedirs(args.path_log, exist_ok=True)
    logger = utils.create_logging(os.path.join(args.path_log, '%s_train.log' % creat_time))

    # get datasets
    train_loader = dataset.load_train_dataset(args)
    val_loader = dataset.load_val_dataset(args)

    # print args
    for param in sorted(vars(args).keys()):
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # get net
    net = models.__dict__[args.model_file].__dict__[args.model_name](logger, args)
    net.cuda()
    net = torch.nn.DataParallel(net)

    # get criterion
    criterion = utils.LabelSmoothingLoss(classes=args.num_classes, smoothing=0.1).cuda()

    # get optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.start_lr, momentum=0.9, weight_decay=1e-4)

    start_epoch = 1
    if args.resume:
        if args.resume in ['best', 'latest']:
            args.resume = os.path.join(args.path_log, 'fold%s_%s.pth' % (args.fold, args.resume))
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            state_dict = torch.load(args.resume)
            if 'model' in state_dict:
                start_epoch = state_dict['epoch'] + 1
                net.load_state_dict(state_dict['model'])
                optimizer.load_state_dict(state_dict['optimizer'])
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, state_dict['epoch']))
            else:
                net.load_state_dict(state_dict)
                logger.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        epoch = start_epoch - 1
        acc1, acc5, outputs = validate(val_loader, net, criterion, epoch, logger, args)
        logger.info('\t'.join(outputs))
        logger.info('Exp path: %s' % args.path_log)
        return

    best_acc1 = 0.0
    best_acc5 = 0.0
    args.time_sec_tot = 0.0
    args.start_epoch = start_epoch
    for epoch in range(start_epoch, args.stop_epoch + 1):
        train(train_loader, net, criterion, optimizer, epoch, logger, args)
        utils.save_checkpoint(epoch, net, optimizer, args, save_name='latest')
        acc1, acc5, outputs = validate(val_loader, net, criterion, epoch, logger, args)
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_acc5 = acc5
            utils.save_checkpoint(epoch, net, optimizer, args, save_name='best')

        outputs += [
            'best_acc1: {:.4f}'.format(best_acc1), 'best_acc5: {:.4f}'.format(best_acc5),
            'Copypaste: {:.4f}, {:.4f}'.format(best_acc1, best_acc5)
        ]
        logger.info('\t'.join(outputs))
        logger.info('Exp path: %s' % args.path_log)


def train(train_loader, net, criterion, optimizer, epoch, logger, args):
    # switch to train mode
    net.train()
    minibatch_count = len(train_loader)
    scaler = torch.cuda.amp.GradScaler()
    tstart = time.time()
    for i, (images, target, location) in enumerate(train_loader):
        # change learning rate
        learning_rate = utils.adjust_learning_rate(optimizer, i, epoch, minibatch_count, args)

        # measure data loading time
        tdata = time.time() - tstart

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        location = location.cuda(non_blocking=True).float()

        images, target_a, target_b, lam, index = utils.mixup(images, target, alpha=0.4)
        location = lam * location + (1 - lam) * location[index]

        # compute output
        with torch.cuda.amp.autocast():
            if args.image_only:
                output = net(images)
            else:
                output = net(images, location)
            loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)

        # measure accuracy and record loss
        acc1, acc5 = lam * utils.accuracy(output, target_a, topk=(1, 5)) + (1 - lam) * utils.accuracy(
            output, target_b, topk=(1, 5))

        # compute gradient and do sgd step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        tend = time.time()
        ttrain = tend - tstart
        tstart = tend

        args.time_sec_tot += ttrain
        time_sec_avg = args.time_sec_tot / ((epoch - args.start_epoch) * minibatch_count + i + 1)
        eta_sec = time_sec_avg * ((args.stop_epoch + 1 - epoch) * minibatch_count - i - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

        outputs = [
            "e: {}/{},{}/{}".format(epoch, args.stop_epoch, i, minibatch_count),
            "{:.2f} mb/s".format(1. / ttrain),
            'eta: {}'.format(eta_str),
            'time: {:.3f}'.format(ttrain),
            'data_time: {:.3f}'.format(tdata),
            'lr: {:.4f}'.format(learning_rate),
            'acc1: {:.4f}'.format(acc1.item()),
            'acc5: {:.4f}'.format(acc5.item()),
            'loss: {:.4f}'.format(loss.item()),
        ]

        if tdata / ttrain > .05:
            outputs += [
                "dp/tot: {:.4f}".format(tdata / ttrain),
            ]

        if i % 20 == 0:
            logger.info('\t'.join(outputs))


def validate(val_loader, net, criterion, epoch, logger, args):
    # switch to evaluate mode
    logger.info('eval epoch {}'.format(epoch))
    net.eval()

    acc1_sum = 0
    acc5_sum = 0
    loss = 0
    valdation_num = 0

    for i, (images, target, location) in enumerate(val_loader):
        # compute output
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        location = location.cuda(non_blocking=True).float()

        with torch.no_grad():
            if args.image_only:
                output = net(images)
            else:
                output = net(images, location)
        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        num = target.size(0)
        valdation_num += num
        acc1_sum += acc1.item() * num
        acc5_sum += acc5.item() * num
        loss += criterion(output, target).item()
        if i % 20 == 0:
            logger.info('iter {}/{}'.format(i, len(val_loader)))

    loss = loss / len(val_loader)
    acc1 = acc1_sum / valdation_num
    acc5 = acc5_sum / valdation_num

    outputs = [
        "val e: {}".format(epoch),
        'acc1: {:.4f}'.format(acc1),
        'acc5: {:.4f}'.format(acc5),
        'loss: {:.4f}'.format(loss),
    ]

    return acc1, acc5, outputs


if __name__ == '__main__':
    main()
