import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    def __init__(self, im_shape):
        super().__init__()
        self.im_shape = im_shape
        grid_x, grid_y = torch.meshgrid([torch.arange(im_shape[1]), torch.arange(im_shape[0])], indexing='xy')
        self.grid = torch.stack([grid_x, grid_y], dim=0)

    def forward(self, im, disp):
        T = self.grid.to(disp.device) + disp  # Transformation fields
        T[:, 0, :, :] = 2 * T[:, 0, :, :] / (self.im_shape[1] - 1) - 1
        T[:, 1, :, :] = 2 * T[:, 1, :, :] / (self.im_shape[0] - 1) - 1
        T = torch.permute(T, [0, 2, 3, 1])
        return F.grid_sample(im, T, mode='bilinear', align_corners=True)


class ImageSynthesis(nn.Module):
    def __init__(self, pe_net):
        super().__init__()
        self.pe_net = pe_net

    def forward(self, xb_im, xb_TI, state):
        assert state in ['train', 'eval'], f'The state should be \'train\' or \'eval\'. Found: \'{state:s}\''

        xb_spl_num, frame_num, im_shape = xb_im.shape[0], xb_im.shape[1], [xb_im.shape[2], xb_im.shape[3]]

        xb_TI = torch.repeat_interleave(xb_TI, dim=0, repeats=im_shape[0] * im_shape[1])

        # Compute the T1 map
        xb_im = torch.permute(xb_im, [0, 2, 3, 1])
        xb_im = xb_im.reshape([xb_spl_num * im_shape[0] * im_shape[1], frame_num])
        xb_im_norm = (torch.sum(xb_im ** 2, dim=1, keepdim=True) + 1e-5) ** 0.5
        xb_im = xb_im / (xb_im_norm + 1e-5)  # Normalization
        xb_im_TI = torch.concatenate([xb_im, xb_TI], dim=1)
        if state == 'train':
            self.pe_net.train()
            xb_T1_map = self.pe_net(xb_im_TI)
        elif state == 'eval':
            self.pe_net.eval()
            with torch.no_grad():
                xb_T1_map = self.pe_net(xb_im_TI)
        xb_T1_map = torch.clamp(xb_T1_map, min=50 / 1000, max=3000 / 1000)

        # Compute the synthetic images
        xb_syn_im = torch.abs(1 - 2 * torch.exp(-xb_TI / xb_T1_map))
        # Magnitude scaling
        xb_syn_im_norm = (torch.sum(xb_syn_im ** 2, dim=1, keepdim=True) + 1e-5) ** 0.5
        xb_syn_im = xb_syn_im / (xb_syn_im_norm + 1e-5)
        xb_syn_im = xb_im_norm * torch.sum(xb_im * xb_syn_im, dim=1, keepdim=True) * xb_syn_im

        xb_T1_map = xb_T1_map.reshape([xb_spl_num, im_shape[0], im_shape[1], 1])
        xb_T1_map = torch.permute(xb_T1_map, [0, 3, 1, 2])
        xb_syn_im = xb_syn_im.reshape([xb_spl_num, im_shape[0], im_shape[1], frame_num])
        xb_syn_im = torch.permute(xb_syn_im, [0, 3, 1, 2])

        return xb_T1_map, xb_syn_im


class ParameterEstimation(nn.Module):
    def __init__(self, pe_net):
        super().__init__()
        self.pe_net = pe_net

    def forward(self, xb_im, xb_TI):
        xb_spl_num, frame_num, im_shape = xb_im.shape[0], xb_im.shape[1], [xb_im.shape[2], xb_im.shape[3]]

        xb_TI = torch.repeat_interleave(xb_TI, dim=0, repeats=im_shape[0] * im_shape[1])

        # Compute the T1 map
        xb_im = torch.permute(xb_im, [0, 2, 3, 1])
        xb_im = xb_im.reshape([xb_spl_num * im_shape[0] * im_shape[1], frame_num])
        xb_im_norm = (torch.sum(xb_im ** 2, dim=1, keepdim=True) + 1e-5) ** 0.5
        xb_im = xb_im / (xb_im_norm + 1e-5)  # Normalization
        xb_im_TI = torch.concatenate([xb_im, xb_TI], dim=1)
        self.pe_net.eval()
        with torch.no_grad():
            xb_T1_map = self.pe_net(xb_im_TI)
        xb_T1_map = torch.clamp(xb_T1_map, min=50 / 1000, max=3000 / 1000)

        xb_T1_map = xb_T1_map.reshape([xb_spl_num, im_shape[0], im_shape[1], 1])
        xb_T1_map = torch.permute(xb_T1_map, [0, 3, 1, 2])

        return xb_T1_map
