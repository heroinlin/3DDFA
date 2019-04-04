# -*- coding: utf-8 -*-

import os


class AverageMeter(object):
    """Modified from Tong Xiao's open-reid.
    Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / (self.count + 1e-20)


class RunningAverageMeter(object):
    """Computes and stores the running average and current value"""

    def __init__(self, hist=0.99):
        self.val = None
        self.avg = None
        self.hist = hist

    def reset(self):
        self.val = None
        self.avg = None

    def update(self, val):
        if self.avg is None:
            self.avg = val
        else:
            self.avg = self.avg * self.hist + val * (1 - self.hist)
        self.val = val


class RecentAverageMeter(object):
    """Stores and computes the average of recent values."""

    def __init__(self, hist_size=100):
        self.hist_size = hist_size
        self.fifo = []
        self.val = 0

    def reset(self):
        self.fifo = []
        self.val = 0

    def update(self, val):
        self.val = val
        self.fifo.append(val)
        if len(self.fifo) > self.hist_size:
            del self.fifo[0]

    @property
    def avg(self):
        assert len(self.fifo) > 0
        return float(sum(self.fifo)) / len(self.fifo)


def may_make_dir(path):
    """
    Args:
      path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
    Note:
      `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
    """
    # This clause has mistakes:
    # if path is None or '':

    if path in [None, '']:
        return
    if not os.path.exists(path):
        os.makedirs(path)


class ReDirectSTD(object):
    """Modified from Tong Xiao's `Logger` in open-reid.
    This class overwrites sys.stdout or sys.stderr, so that console logs can
    also be written to file.
    Args:
      fpath: file path
      console: one of ['stdout', 'stderr']
      immediately_visible: If `False`, the file is opened only once and closed
        after exiting. In this case, the message written to file may not be
        immediately visible (Because the file handle is occupied by the
        program?). If `True`, each writing operation of the console will
        open, write to, and close the file. If your program has tons of writing
        operations, the cost of opening and closing file may be obvious. (?)
    Usage example:
      `ReDirectSTD('stdout.txt', 'stdout', False)`
      `ReDirectSTD('stderr.txt', 'stderr', False)`
    NOTE: File will be deleted if already existing. Log dir and file is created
      lazily -- if no message is written, the dir and file will not be created.
    """

    def __init__(self, fpath=None, console='stdout', immediately_visible=False):
        import sys
        import os
        import os.path as osp

        assert console in ['stdout', 'stderr']
        self.console = sys.stdout if console == 'stdout' else sys.stderr
        self.file = fpath
        self.f = None
        self.immediately_visible = immediately_visible
        if fpath is not None:
            # Remove existing log file.
            if osp.exists(fpath):
                os.remove(fpath)

        # Overwrite
        if console == 'stdout':
            sys.stdout = self
        else:
            sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            may_make_dir(os.path.dirname(os.path.abspath(self.file)))
            if self.immediately_visible:
                with open(self.file, 'a') as f:
                    f.write(msg)
            else:
                if self.f is None:
                    self.f = open(self.file, 'w')
                self.f.write(msg)

    def flush(self):
        self.console.flush()
        if self.f is not None:
            self.f.flush()
            import os
            os.fsync(self.f.fileno())

    def close(self):
        self.console.close()
        if self.f is not None:
            self.f.close()


def set_seed(seed):
    import random
    random.seed(seed)
    print('setting random-seed to {}'.format(seed))

    import numpy as np
    np.random.seed(seed)
    print('setting np-random-seed to {}'.format(seed))

    import torch
    torch.backends.cudnn.enabled = False
    print('cudnn.enabled set to {}'.format(torch.backends.cudnn.enabled))
    # set seed for CPU
    torch.manual_seed(seed)
    print('setting torch-seed to {}'.format(seed))


def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    """Decay exponentially in the later phase of training. All parameters in the
    optimizer share the same learning rate.

    Args:
      optimizer: a pytorch `Optimizer` object
      base_lr: starting learning rate
      ep: current epoch, ep >= 1
      total_ep: total number of epochs to train
      start_decay_at_ep: start decaying at the BEGINNING of this epoch

    Example:
      base_lr = 2e-4
      total_ep = 300
      start_decay_at_ep = 201
      It means the learning rate starts at 2e-4 and begins decaying after 200
      epochs. And training stops after 300 epochs.

    NOTE:
      It is meant to be called at the BEGINNING of an epoch.
    """
    assert ep >= 1, "Current epoch number should be >= 1"
    if ep < start_decay_at_ep:
        return
    for g in optimizer.param_groups:
        g['lr'] = (base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                        / (total_ep + 1 - start_decay_at_ep))))
        print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


def find_index(seq, item):
    for i, x in enumerate(seq):
        if item == x:
            return i
    return -1


def adjust_lr_staircase(param_groups, base_lrs, ep, decay_at_epochs, factor):
    """Multiplied by a factor at the BEGINNING of specified epochs. Different
    param groups specify their own base learning rates.

    Args:
      param_groups: a list of params
      base_lrs: starting learning rates, len(base_lrs) = len(param_groups)
      ep: current epoch, ep >= 1
      decay_at_epochs: a list or tuple; learning rates are multiplied by a factor
        at the BEGINNING of these epochs
      factor: a number in range (0, 1)

    Example:
      base_lrs = [0.1, 0.01]
      decay_at_epochs = [51, 101]
      factor = 0.1
      It means the learning rate starts at 0.1 for 1st param group
      (0.01 for 2nd param group) and is multiplied by 0.1 at the
      BEGINNING of the 51'st epoch, and then further multiplied by 0.1 at the
      BEGINNING of the 101'st epoch, then stays unchanged till the end of
      training.

    NOTE:
      It is meant to be called at the BEGINNING of an epoch.
    """
    assert len(base_lrs) == len(param_groups), \
        "You should specify base lr for each param group."
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep not in decay_at_epochs:
        return

    ind = find_index(decay_at_epochs, ep)
    for i, (g, base_lr) in enumerate(zip(param_groups, base_lrs)):
        g['lr'] = base_lr * factor ** (ind + 1)
        print('=====> Param group {}: lr adjusted to {:.10f}'
              .format(i, g['lr']).rstrip('0'))
