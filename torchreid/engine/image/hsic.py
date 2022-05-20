from __future__ import division, print_function, absolute_import

import json
import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter




from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss

from ..engine import Engine

from torchreid import metrics
from torchreid.utils import (
    MetricMeter, AverageMeter, re_ranking, open_all_layers, save_checkpoint,
    open_specified_layers, visualize_ranked_results
)


class ImageHsicEngine(Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        bias_model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        weight_h=0.1,
        hsic_factor = 0.5,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        alternative = False
    ):
        super(ImageHsicEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.bias_model = bias_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.hsic_factor = hsic_factor
        self.alternative = alternative
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0
        assert weight_t + weight_x > 0
        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_h = weight_h

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        
    
    
    def _kernel(self, X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)

        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX


    def hsic_loss(self, input1, input2, unbiased=False):
        N = len(input1)
        if N < 4:
            return torch.tensor(0.0).to(input1.device)
        # we simply use the squared dimension of feature as the sigma for RBF kernel
        sigma_x = np.sqrt(input1.size()[1])
        sigma_y = np.sqrt(input2.size()[1])

        # compute the kernels
        kernel_XX = self._kernel(input1, sigma_x)
        kernel_YY = self._kernel(input2, sigma_y)

        if unbiased:
            """Unbiased estimator of Hilbert-Schmidt Independence Criterion
            Song, Le, et al. "Feature selection via dependence maximization." 2012.
            """
            tK = kernel_XX - torch.diag(kernel_XX)
            tL = kernel_YY - torch.diag(kernel_YY)
            hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
            )
            loss = hsic / (N * (N - 3))
        else:
            """Biased estimator of Hilbert-Schmidt Independence Criterion
            Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
            """
            KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
            LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
            loss = torch.trace(KH @ LH / (N - 1) ** 2)
        return loss
        
    

    def forward_backward(self, data ,_iter):
        imgs, bias_imgs ,pids = self.parse_data_for_train_hsic(data)
        
        if self.use_gpu:
            imgs = imgs.cuda()
            bias_imgs = bias_imgs.cuda()
            pids = pids.cuda()
            

        outputs, feat_unbias = self.model(imgs)
        outputs_bias, feat_bias = self.bias_model(bias_imgs)

        loss = 0
        loss_hsic_f=0
        loss_hsic_g=0
        loss_summary = {}
        
        

        if self.weight_t > 0:
            loss_t_unbias = self.compute_loss(self.criterion_t, feat_unbias, pids)
            loss_t_bias = self.compute_loss(self.criterion_t, feat_bias, pids)
            loss += self.weight_t * loss_t_unbias
            loss += self.weight_t * loss_t_bias
            loss_summary['loss_t_unbias'] = loss_t_unbias.item()
            loss_summary['loss_t_bias'] = loss_t_bias.item()

        if self.weight_x > 0:
            loss_x_unbias = self.compute_loss(self.criterion_x, outputs, pids)
            loss_x_bias = self.compute_loss(self.criterion_x, outputs_bias, pids)
            loss += self.weight_x * loss_x_unbias
            loss += self.weight_x * loss_x_bias
            loss_summary['loss_x_unbias'] = loss_x_unbias.item()
            loss_summary['loss_x_bias'] = loss_x_bias.item()
            loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()
            
        loss_hsic_f += self.hsic_factor * self.hsic_loss(feat_unbias, feat_bias.detach(), unbiased=True) 
        loss_hsic_g += - self.hsic_factor * self.hsic_loss(feat_unbias.detach(), feat_bias, unbiased=True)
        
        
        loss_summary['loss_hsic_f'] = loss_hsic_f.item()
        loss_summary['loss_hsic_g'] = loss_hsic_g.item()
        
        
        loss_mask = _iter % 2 
        loss_hsic = loss_mask * loss_hsic_f + (1-loss_mask) * loss_hsic_g
        loss += self.weight_h * loss_hsic
        
        
        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary
    
    def train(self, print_freq=10, fixbase_epoch=0, open_layers=None):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.set_model_mode('train')

        self.two_stepped_transfer_learning(
            self.epoch, fixbase_epoch, open_layers
        )

        self.num_batches = len(self.train_loader)
        end = time.time()
        for self.batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(data, _iter=self.batch_idx )
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % print_freq == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch: [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr:.6f}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta_str,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            if self.writer is not None:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/data', data_time.avg, n_iter)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar('Train/' + name, meter.avg, n_iter)
                self.writer.add_scalar(
                    'Train/lr', self.get_current_lr(), n_iter
                )

            end = time.time()

        self.update_lr()
