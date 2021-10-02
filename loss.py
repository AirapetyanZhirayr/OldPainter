import torch

import torch
import torch.nn as nn
import torchvision
import utils
import matplotlib.pyplot as plt
import numpy as np
import random
import pytorch_batch_sinkhorn as spc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PixelLoss(nn.Module):

    def __init__(self, p=1):
        super(PixelLoss, self).__init__()
        self.p = p

    def forward(self, canvas, gt, ignore_color=False):
        if ignore_color:
            canvas = torch.mean(canvas, dim=1)
            gt = torch.mean(gt, dim=1)
        loss = torch.mean(torch.abs(canvas-gt)**self.p)
        return loss


class SinkhornLoss(nn.Module):

    def __init__(self, epsilon=0.01, niter=5, normalize=False):
        super(SinkhornLoss, self).__init__()
        self.epsilon = epsilon
        self.niter = niter
        self.normalize = normalize

    def _mesh_grids(self, batch_size, h, w):

        a = torch.linspace(0.0, h - 1.0, h).to(device)
        b = torch.linspace(0.0, w - 1.0, w).to(device)
        y_grid = a.view(-1, 1).repeat(batch_size, 1, w) / h
        x_grid = b.view(1, -1).repeat(batch_size, h, 1) / w
        grids = torch.cat([y_grid.view(batch_size, -1, 1), x_grid.view(batch_size, -1, 1)], dim=-1)
        return grids

    def forward(self, canvas, gt):

        batch_size, c, h, w = gt.shape
        if h > 24:
            canvas = nn.functional.interpolate(canvas, [24, 24], mode='area')
            gt = nn.functional.interpolate(gt, [24, 24], mode='area')
            batch_size, c, h, w = gt.shape

        canvas_grids = self._mesh_grids(batch_size, h, w)
        gt_grids = torch.clone(canvas_grids)

        # randomly select a color channel, to speedup and consume memory
        i = random.randint(0, 2)

        img_1 = canvas[:, [i], :, :]
        img_2 = gt[:, [i], :, :]

        mass_x = img_1.reshape(batch_size, -1)
        mass_y = img_2.reshape(batch_size, -1)
        if self.normalize:
            loss = spc.sinkhorn_normalized(
                canvas_grids, gt_grids, epsilon=self.epsilon, niter=self.niter,
                mass_x=mass_x, mass_y=mass_y)
        else:
            loss = spc.sinkhorn_loss(
                canvas_grids, gt_grids, epsilon=self.epsilon, niter=self.niter,
                mass_x=mass_x, mass_y=mass_y)


        return loss
