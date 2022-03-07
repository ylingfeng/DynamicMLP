#!/usr/bin/env python3
import logging
import os

import numpy as np
import torch
import torch.nn as nn


def get_flops(model, logger=None, loc_cin=6, img_size=224, image_only=False):
    from thop import clever_format, profile
    bs = 2
    img = torch.randn(bs, 3, img_size, img_size)
    loc = torch.randn(bs, loc_cin)
    if image_only:
        flops, params = profile(model, inputs=(img, ))
    else:
        flops, params = profile(model, inputs=(img, loc))
    flops = flops / bs
    flops, params = clever_format([flops, params], "%.3f")

    if logger is not None:
        logger.info('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))


def mixup(x, y, alpha=0.4):
    """
    mixup: Beyond Empirical Risk Minimization.ICLR 2018
    https://arxiv.org/pdf/1710.09412.pdf
    https://github.com/facebookresearch/mixup-cifar10

    Args:
        Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam, index


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def create_logging(log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger()

    handlers = []
    stream_handler = logging.StreamHandler()
    handlers.append(stream_handler)

    rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    return logger


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return np.array(res)


def save_checkpoint(epoch, model, optimizer, args, save_name='latest'):
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state_dict, os.path.join(args.path_log, 'fold%s_%s.pth' % (args.fold, save_name)))


def adjust_learning_rate(optimizer, idx, epoch, minibatch_count, args):
    # epoch >= 1
    if epoch <= args.warmup:
        lr = args.start_lr * ((epoch - 1) / args.warmup + idx / (args.warmup * minibatch_count))
    else:
        decay_rate = 0.5 * (1 + np.cos((epoch - 1) * np.pi / args.stop_epoch))
        lr = args.start_lr * decay_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
