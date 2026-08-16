import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NMAE(nn.Module):
    # Normalized mean absolute error (2D or 3D)
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.abs((x - y) / (y + 1e-5)))


class LNCC(nn.Module):
    # Local normalized cross-correlation (2D or 3D)
    def __init__(self, win_shape=None):
        super().__init__()
        self.win_shape = win_shape

    def forward(self, x, y):
        Ii = x
        Ji = y

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, 1, *im_shape]
        dim_num = len(list(Ii.shape)) - 2
        assert dim_num in [1, 2, 3], f'The input images should be 1D to 3D. Found: {dim_num:d}'

        # set window shape
        win_shape = [9] * dim_num if self.win_shape is None else self.win_shape

        # compute filters
        sum_filt = torch.ones([1, 1, *win_shape]).to(x.device)

        pad_len = math.floor(max(win_shape) / 2)

        if dim_num == 1:
            stride = (1,)
            padding = (pad_len,)
        elif dim_num == 2:
            stride = (1, 1)
            padding = (pad_len, pad_len)
        else:
            stride = (1, 1, 1)
            padding = (pad_len, pad_len, pad_len)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % dim_num)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        n = math.prod(win_shape)
        u_I = I_sum / n
        u_J = J_sum / n

        cross = IJ_sum - n * u_J * u_I
        I_var = I2_sum - n * u_I * u_I
        J_var = J2_sum - n * u_J * u_J

        cc = cross * cross / (I_var * J_var + 1e-5)

        return 1 - torch.mean(cc)


class ComputeSingularValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        # X shape: [B,M,N]
        # s shape: [B,min(M,N)]
        U, s, VH = torch.linalg.svd(X, full_matrices=False)
        ctx.save_for_backward(U, VH)

        return s

    @staticmethod
    def backward(ctx, grad_s):
        # grad_s shape: [B,min(M,N)]
        U, VH = ctx.saved_tensors

        return torch.matmul(U * torch.unsqueeze(grad_s, dim=1), VH)


class LR(nn.Module):
    # Low-rank constraint (2D or 3D)
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: [B,Nt,Ny,Nx] or [B,Nt,Ny,Nx,Nz]
        x_shape = torch.tensor(x.shape)
        B, M, N = x_shape[0], x_shape[1], torch.prod(x_shape[2:])
        x = x.reshape([B, M, N])
        s = ComputeSingularValues.apply(x)

        return torch.mean(s[:, 2:])  # Expected rank: 2


class GDS(nn.Module):
    # Generalized Dice score (2D or 3D)
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        sum_dims = list(range(2, x.ndim))
        w = 1 / (torch.sum(y, dim=sum_dims) ** 2 + 1e-5)
        xy_sum = torch.sum(x * y, dim=sum_dims)
        xx_sum = torch.sum(x * x, dim=sum_dims)
        yy_sum = torch.sum(y * y, dim=sum_dims)
        num = 2 * torch.sum(w * xy_sum, dim=1)
        den = torch.sum(w * (xx_sum + yy_sum), dim=1)
        dice = torch.mean(num / (den + 1e-5))

        return 1 - dice


class SpatialRegularizer(nn.Module):
    # First/second order regularizer for spatially smooth deformation (only 2D)
    def __init__(self, order):
        super().__init__()
        assert order in [1, 2], f'The regularizer should be first or second order. Found: {order:d}'
        self.order = order

    def forward(self, vel):
        if self.order == 1:
            # Diffusion regularizer
            dx = vel[:, :, :, 1:] - vel[:, :, :, :-1]
            dy = vel[:, :, 1:, :] - vel[:, :, :-1, :]
            dx2 = dx * dx
            dy2 = dy * dy
            r = torch.mean(dx2) + torch.mean(dy2)
        else:
            # Bending energy regularizer
            dx = vel[:, :, :, 1:] - vel[:, :, :, :-1]
            dy = vel[:, :, 1:, :] - vel[:, :, :-1, :]
            dxx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
            dyy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
            dxy = dx[:, :, 1:, :] - dx[:, :, :-1, :]
            dxx2 = dxx * dxx
            dyy2 = dyy * dyy
            dxy2 = dxy * dxy
            r = torch.mean(dxx2) + 2 * torch.mean(dxy2) + torch.mean(dyy2)

        return r
