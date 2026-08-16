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
    def __init__(self, pe_net, ss_net):
        super().__init__()
        self.pe_net = pe_net
        self.ss_net = ss_net

    def forward(self, xb_im, xb_RR, state):
        assert state in ['train', 'eval'], f'The state should be \'train\' or \'eval\'. Found: \'{state:s}\''

        xb_spl_num, frame_num, im_shape = xb_im.shape[0], xb_im.shape[1], [xb_im.shape[2], xb_im.shape[3]]

        xb_RR = torch.repeat_interleave(xb_RR, dim=0, repeats=im_shape[0] * im_shape[1])

        # Compute the parametric maps
        xb_im = torch.permute(xb_im, [0, 2, 3, 1])
        xb_im = xb_im.reshape([xb_spl_num * im_shape[0] * im_shape[1], frame_num])
        xb_im_norm = (torch.sum(xb_im ** 2, dim=1, keepdim=True) + 1e-5) ** 0.5
        xb_im = xb_im / (xb_im_norm + 1e-5)  # Normalization
        xb_im_RR = torch.concatenate([xb_im, xb_RR], dim=1)
        if state == 'train':
            self.pe_net.train()
            xb_map = self.pe_net(xb_im_RR)
        elif state == 'eval':
            self.pe_net.eval()
            with torch.no_grad():
                xb_map = self.pe_net(xb_im_RR)
        xb_T1_map = torch.clamp(xb_map[:, 0:1], min=50 / 1000, max=3000 / 1000)
        xb_T2_map = torch.clamp(xb_map[:, 1:2], min=5 / 1000, max=250 / 1000)
        xb_map = torch.concatenate([xb_T1_map, xb_T2_map], dim=1)

        # Compute the synthetic images
        xb_map_RR = torch.concatenate([xb_map, xb_RR], dim=1)
        self.ss_net.eval()
        xb_syn_im = self.ss_net(xb_map_RR)
        # Magnitude scaling
        xb_syn_im_norm = (torch.sum(xb_syn_im ** 2, dim=1, keepdim=True) + 1e-5) ** 0.5
        xb_syn_im = xb_syn_im / (xb_syn_im_norm + 1e-5)
        xb_syn_im = xb_im_norm * torch.sum(xb_im * xb_syn_im, dim=1, keepdim=True) * xb_syn_im

        xb_map = xb_map.reshape([xb_spl_num, im_shape[0], im_shape[1], xb_map.shape[1]])
        xb_map = torch.permute(xb_map, [0, 3, 1, 2])
        xb_syn_im = xb_syn_im.reshape([xb_spl_num, im_shape[0], im_shape[1], frame_num])
        xb_syn_im = torch.permute(xb_syn_im, [0, 3, 1, 2])

        return xb_map, xb_syn_im


class ParameterEstimation(nn.Module):
    def __init__(self, pe_net):
        super().__init__()
        self.pe_net = pe_net

    def forward(self, xb_im, xb_RR):
        xb_spl_num, frame_num, im_shape = xb_im.shape[0], xb_im.shape[1], [xb_im.shape[2], xb_im.shape[3]]

        xb_RR = torch.repeat_interleave(xb_RR, dim=0, repeats=im_shape[0] * im_shape[1])

        # Compute the parametric maps
        xb_im = torch.permute(xb_im, [0, 2, 3, 1])
        xb_im = xb_im.reshape([xb_spl_num * im_shape[0] * im_shape[1], frame_num])
        xb_im_norm = (torch.sum(xb_im ** 2, dim=1, keepdim=True) + 1e-5) ** 0.5
        xb_im = xb_im / (xb_im_norm + 1e-5)  # Normalization
        xb_im_RR = torch.concatenate([xb_im, xb_RR], dim=1)
        self.pe_net.eval()
        with torch.no_grad():
            xb_map = self.pe_net(xb_im_RR)
        xb_T1_map = torch.clamp(xb_map[:, 0:1], min=50 / 1000, max=3000 / 1000)
        xb_T2_map = torch.clamp(xb_map[:, 1:2], min=5 / 1000, max=250 / 1000)
        xb_map = torch.concatenate([xb_T1_map, xb_T2_map], dim=1)

        xb_map = xb_map.reshape([xb_spl_num, im_shape[0], im_shape[1], xb_map.shape[1]])
        xb_map = torch.permute(xb_map, [0, 3, 1, 2])

        return xb_map
