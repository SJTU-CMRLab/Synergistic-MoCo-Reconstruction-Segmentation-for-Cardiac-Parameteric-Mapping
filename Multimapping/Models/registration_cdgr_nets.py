import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CDGRSpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fea, disp):
        # fea shape: [B,C,Nt,Ny,Nx]
        # disp shape: [B,2,Nt,Ny,Nx]
        frame_num = fea.shape[2]  # Nt
        fea_shape = fea.shape[3:]  # [Ny,Nx]

        grid_x, grid_y = torch.meshgrid([torch.arange(fea_shape[1]), torch.arange(fea_shape[0])], indexing='xy')
        grid = torch.stack([grid_x, grid_y], dim=0)  # [2,Ny,Nx]
        grid = torch.unsqueeze(grid, dim=1)  # [2,1,Ny,Nx]
        T = grid.to(disp.device) + disp  # Transformation fields, [B,2,Nt,Ny,Nx]
        T[:, 0, :, :, :] = 2 * T[:, 0, :, :, :] / (fea_shape[1] - 1) - 1
        T[:, 1, :, :, :] = 2 * T[:, 1, :, :, :] / (fea_shape[0] - 1) - 1
        T = torch.permute(T, [0, 2, 3, 4, 1])  # [B,Nt,Ny,Nx,2]

        new_fea = []
        for t in range(frame_num):
            tmp = F.grid_sample(fea[:, :, t, :, :], T[:, t, :, :, :], mode='bilinear',
                                align_corners=True)  # [B,C,Ny,Nx]
            new_fea.append(tmp)
        new_fea = torch.stack(new_fea, dim=2)  # [B,C,Nt,Ny,Nx]

        return new_fea


class CDGRVelCon(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, vel):
        # vel shape: [B,2,Nt,Ny,Nx]
        mean_vel_x = torch.mean(vel[:, 0:1, :, :, :], dim=2, keepdim=True)  # [B,1,1,Ny,Nx]
        mean_vel_y = torch.mean(vel[:, 1:2, :, :, :], dim=2, keepdim=True)  # [B,1,1,Ny,Nx]
        new_vel_x = vel[:, 0:1, :, :, :] - mean_vel_x  # [B,1,Nt,Ny,Nx]
        new_vel_y = vel[:, 1:2, :, :, :] - mean_vel_y  # [B,1,Nt,Ny,Nx]

        return torch.concatenate([new_vel_x, new_vel_y], dim=1)  # [B,2,Nt,Ny,Nx]


class CDGRVelInt(nn.Module):
    def __init__(self, step_num):
        super().__init__()
        self.step_num = step_num
        self.scaling = 1 / (2 ** step_num)
        self.spatial_transformer = CDGRSpatialTransformer()

    def forward(self, vel):
        # vel shape: [B,2,Nt,Ny,Nx]
        disp = self.scaling * vel
        for _ in range(self.step_num):
            disp = disp + self.spatial_transformer(disp, disp)
        return disp  # [B,2,Nt,Ny,Nx]


class CDGRVelResize(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, vel):
        # vel shape: [B,2,Nt,Ny,Nx]
        frame_num = vel.shape[2]  # Nt
        im_shape = vel.shape[3:]  # [Ny,Nx]
        new_vel = []
        for t in range(frame_num):
            tmp = F.interpolate(vel[:, :, t, :, :], scale_factor=self.factor, mode='bilinear',
                                align_corners=True)  # [B,2,factor*Ny,factor*Nx]
            tmp[:, 0, :, :] = tmp[:, 0, :, :] * (self.factor * im_shape[1] - 1) / (im_shape[1] - 1)
            tmp[:, 1, :, :] = tmp[:, 1, :, :] * (self.factor * im_shape[0] - 1) / (im_shape[0] - 1)
            new_vel.append(tmp)
        new_vel = torch.stack(new_vel, dim=2)  # [B,2,Nt,factor*Ny,factor*Nx]

        return new_vel


class SpaceTimeSelfAttention(nn.Module):
    # Space-time self-attention block without embedding
    def __init__(self, emb_dim, head_num, drop_rate):
        super().__init__()

        # Spatial multi-head attention
        self.layer_norm_spa_mha = nn.LayerNorm(emb_dim)
        self.spa_multi_head_attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=head_num,
                                                              batch_first=True, dropout=drop_rate)
        self.dropout_spa_mha = nn.Dropout(drop_rate)

        # Temporal multi-head attention
        self.layer_norm_tem_mha = nn.LayerNorm(emb_dim)
        self.tem_multi_head_attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=head_num,
                                                              batch_first=True, dropout=drop_rate)
        self.dropout_tem_mha = nn.Dropout(drop_rate)

        # Multi-layer perceptron (GEGLU)
        self.layer_norm_mlp = nn.LayerNorm(emb_dim)
        self.linear_mlp11 = nn.Linear(emb_dim, 2 * emb_dim)
        self.linear_mlp12 = nn.Linear(emb_dim, 2 * emb_dim)
        self.gelu = nn.GELU()
        self.dropout_mlp1 = nn.Dropout(drop_rate)
        self.linear_mlp2 = nn.Linear(2 * emb_dim, emb_dim)
        self.dropout_mlp2 = nn.Dropout(drop_rate)

    def forward(self, x):
        # x shape: [B,emb_dim,Nt,Ny,Nx]
        B, _, _, Ny, _ = x.shape

        # Spatial sequence rearrangement
        x = rearrange(x, 'B C Nt Ny Nx -> (B Nt) (Ny Nx) C')  # [B*Nt,Ny*Nx,emb_dim]

        # Spatial multi-head attention
        x_spa_mha = self.layer_norm_spa_mha(x)  # [B*Nt,Ny*Nx,emb_dim]
        x_spa_mha, _ = self.spa_multi_head_attention(x_spa_mha, x_spa_mha, x_spa_mha,
                                                     need_weights=False)  # [B*Nt,Ny*Nx,emb_dim]
        x_spa_mha = self.dropout_spa_mha(x_spa_mha)  # [B*Nt,Ny*Nx,emb_dim]
        x_spa_mha = x_spa_mha + x  # [B*Nt,Ny*Nx,emb_dim]

        # Temporal sequence rearrangement
        x = rearrange(x_spa_mha, '(B Nt) (Ny Nx) C -> (B Ny Nx) Nt C', B=B, Ny=Ny)  # [B*Ny*Nx,Nt,emb_dim]

        # Temporal multi-head attention
        x_tem_mha = self.layer_norm_tem_mha(x)  # [B*Ny*Nx,Nt,emb_dim]
        x_tem_mha, _ = self.tem_multi_head_attention(x_tem_mha, x_tem_mha, x_tem_mha,
                                                     need_weights=False)  # [B*Ny*Nx,Nt,emb_dim]
        x_tem_mha = self.dropout_tem_mha(x_tem_mha)  # [B*Ny*Nx,Nt,emb_dim]
        x_tem_mha = x_tem_mha + x  # [B*Ny*Nx,Nt,emb_dim]

        # Multi-layer perceptron (GEGLU)
        x_mlp = self.layer_norm_mlp(x_tem_mha)  # [B*Ny*Nx,Nt,emb_dim]
        x_mlp = self.gelu(self.linear_mlp11(x_mlp)) * self.linear_mlp12(x_mlp)  # [B*Ny*Nx,Nt,2*emb_dim]
        x_mlp = self.dropout_mlp1(x_mlp)  # [B*Ny*Nx,Nt,2*emb_dim]
        x_mlp = self.linear_mlp2(x_mlp)  # [B*Ny*Nx,Nt,emb_dim]
        x_mlp = self.dropout_mlp2(x_mlp)  # [B*Ny*Nx,Nt,emb_dim]
        x_mlp = x_mlp + x_tem_mha  # [B*Ny*Nx,Nt,emb_dim]

        # Spatiotemporal image rearrangement
        x_mlp = rearrange(x_mlp, '(B Ny Nx) Nt C -> B C Nt Ny Nx', B=B, Ny=Ny)  # [B,emb_dim,Nt,Ny,Nx]

        return x_mlp


def build_conv2d_plus_t_block(in_ch_num, out_ch_num1, out_ch_num2):
    return nn.Sequential(
        nn.Conv3d(in_ch_num, out_ch_num1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='zeros'),
        nn.GroupNorm(num_groups=16, num_channels=out_ch_num1),
        nn.LeakyReLU(),
        nn.Conv3d(out_ch_num1, out_ch_num1, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0),
                  padding_mode='circular'),
        nn.GroupNorm(num_groups=16, num_channels=out_ch_num1),
        nn.LeakyReLU(),
        nn.Conv3d(out_ch_num1, out_ch_num2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='zeros'),
        nn.GroupNorm(num_groups=16, num_channels=out_ch_num2),
        nn.LeakyReLU(),
        nn.Conv3d(out_ch_num2, out_ch_num2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0),
                  padding_mode='circular'),
        nn.GroupNorm(num_groups=16, num_channels=out_ch_num2),
        nn.LeakyReLU()
    )


def build_conv2d_plus_t_reg_head(in_ch_num):
    return nn.Sequential(
        nn.Conv3d(in_ch_num, in_ch_num, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='zeros'),
        nn.GroupNorm(num_groups=16, num_channels=in_ch_num),
        nn.LeakyReLU(),
        nn.Conv3d(in_ch_num, 2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), padding_mode='circular')
    )


class CDGRNet(nn.Module):
    # CDGR-Net with space-time convolution and space-time self-attention
    def __init__(self, im_shape, frame_num, ch_num, drop_rate):
        super().__init__()

        # Layers
        self.enc_block1 = build_conv2d_plus_t_block(1, ch_num, ch_num)
        self.enc_block2 = build_conv2d_plus_t_block(ch_num, 2 * ch_num, 2 * ch_num)
        self.enc_block3 = build_conv2d_plus_t_block(2 * ch_num, 4 * ch_num, 4 * ch_num)
        self.enc_block4 = build_conv2d_plus_t_block(4 * ch_num, 8 * ch_num, 8 * ch_num)
        self.bridge_block = build_conv2d_plus_t_block(8 * ch_num, 16 * ch_num, 16 * ch_num)

        self.patch_to_emb = nn.Conv3d(16 * ch_num, 8 * ch_num, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                      padding=(0, 0, 0))
        self.spa_pos_enc = nn.Parameter(torch.randn((1, 8 * ch_num, 1, int(im_shape[0] / 16), int(im_shape[1] / 16))))
        self.tem_pos_enc = nn.Parameter(torch.randn((1, 8 * ch_num, frame_num, 1, 1)))
        self.sa_block1 = SpaceTimeSelfAttention(emb_dim=8 * ch_num, head_num=int(ch_num / 2), drop_rate=drop_rate)
        self.sa_block2 = SpaceTimeSelfAttention(emb_dim=8 * ch_num, head_num=int(ch_num / 2), drop_rate=drop_rate)
        self.sa_block3 = SpaceTimeSelfAttention(emb_dim=8 * ch_num, head_num=int(ch_num / 2), drop_rate=drop_rate)
        self.sa_block4 = SpaceTimeSelfAttention(emb_dim=8 * ch_num, head_num=int(ch_num / 2), drop_rate=drop_rate)
        self.sa_block5 = SpaceTimeSelfAttention(emb_dim=8 * ch_num, head_num=int(ch_num / 2), drop_rate=drop_rate)
        self.sa_block6 = SpaceTimeSelfAttention(emb_dim=8 * ch_num, head_num=int(ch_num / 2), drop_rate=drop_rate)

        self.dec_block4 = build_conv2d_plus_t_block(16 * ch_num, 8 * ch_num, 4 * ch_num)
        self.dec_block3 = build_conv2d_plus_t_block(8 * ch_num, 4 * ch_num, 2 * ch_num)
        self.dec_block2 = build_conv2d_plus_t_block(4 * ch_num, 2 * ch_num, ch_num)
        self.dec_block1 = build_conv2d_plus_t_block(2 * ch_num, ch_num, ch_num)

        self.reg_head_bridge = build_conv2d_plus_t_reg_head(8 * ch_num)
        self.reg_head_bridge[-1].weight = nn.Parameter(1e-5 * torch.randn(self.reg_head_bridge[-1].weight.shape))
        self.reg_head_bridge[-1].bias = nn.Parameter(torch.zeros(self.reg_head_bridge[-1].bias.shape))
        self.reg_head_dec4 = build_conv2d_plus_t_reg_head(4 * ch_num)
        self.reg_head_dec4[-1].weight = nn.Parameter(1e-5 * torch.randn(self.reg_head_dec4[-1].weight.shape))
        self.reg_head_dec4[-1].bias = nn.Parameter(torch.zeros(self.reg_head_dec4[-1].bias.shape))
        self.reg_head_dec3 = build_conv2d_plus_t_reg_head(2 * ch_num)
        self.reg_head_dec3[-1].weight = nn.Parameter(1e-5 * torch.randn(self.reg_head_dec3[-1].weight.shape))
        self.reg_head_dec3[-1].bias = nn.Parameter(torch.zeros(self.reg_head_dec3[-1].bias.shape))
        self.reg_head_dec2 = build_conv2d_plus_t_reg_head(ch_num)
        self.reg_head_dec2[-1].weight = nn.Parameter(1e-5 * torch.randn(self.reg_head_dec2[-1].weight.shape))
        self.reg_head_dec2[-1].bias = nn.Parameter(torch.zeros(self.reg_head_dec2[-1].bias.shape))
        self.reg_head_dec1 = build_conv2d_plus_t_reg_head(ch_num)
        self.reg_head_dec1[-1].weight = nn.Parameter(1e-5 * torch.randn(self.reg_head_dec1[-1].weight.shape))
        self.reg_head_dec1[-1].bias = nn.Parameter(torch.zeros(self.reg_head_dec1[-1].bias.shape))

        self.fea_down = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.fea_up = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.spatial_transformer = CDGRSpatialTransformer()
        self.vel_con = CDGRVelCon()
        self.vel_int = CDGRVelInt(step_num=8)
        self.vel_up = CDGRVelResize(factor=2)

    def forward(self, x):
        # x shape: [B,Nt,Ny,Nx]
        x = torch.unsqueeze(x, dim=1)  # [B,1,Nt,Ny,Nx]

        x_enc1 = self.enc_block1(x)  # [B,ch_num,Nt,Ny,Nx]
        x_enc2 = self.enc_block2(self.fea_down(x_enc1))  # [B,2*ch_num,Nt,Ny/2,Nx/2]
        x_enc3 = self.enc_block3(self.fea_down(x_enc2))  # [B,4*ch_num,Nt,Ny/4,Nx/4]
        x_enc4 = self.enc_block4(self.fea_down(x_enc3))  # [B,8*ch_num,Nt,Ny/8,Nx/8]
        x_bridge = self.bridge_block(self.fea_down(x_enc4))  # [B,16*ch_num,Nt,Ny/16,Nx/16]

        # Patch embedding
        x_bridge = self.patch_to_emb(x_bridge)  # [B,8*ch_num,Nt,Ny/16,Nx/16]

        # Positional encoding
        x_bridge = x_bridge + self.spa_pos_enc + self.tem_pos_enc  # [B,8*ch_num,Nt,Ny/16,Nx/16]

        # Space-time self-attention
        x_bridge = self.sa_block1(x_bridge)  # [B,8*ch_num,Nt,Ny/16,Nx/16]
        x_bridge = self.sa_block2(x_bridge)  # [B,8*ch_num,Nt,Ny/16,Nx/16]
        x_bridge = self.sa_block3(x_bridge)  # [B,8*ch_num,Nt,Ny/16,Nx/16]
        x_bridge = self.sa_block4(x_bridge)  # [B,8*ch_num,Nt,Ny/16,Nx/16]
        x_bridge = self.sa_block5(x_bridge)  # [B,8*ch_num,Nt,Ny/16,Nx/16]
        x_bridge = self.sa_block6(x_bridge)  # [B,8*ch_num,Nt,Ny/16,Nx/16]

        # Step 1
        vel = self.vel_con(self.reg_head_bridge(x_bridge))  # [B,2,Nt,Ny/16,Nx/16]

        vel = self.vel_up(vel)  # [B,2,Nt,Ny/8,Nx/8]
        disp = self.vel_int(vel)  # [B,2,Nt,Ny/8,Nx/8]
        def_x_enc4 = self.spatial_transformer(x_enc4, disp)  # [B,8*ch_num,Nt,Ny/8,Nx/8]

        # Step 2
        x_dec4 = self.dec_block4(
            torch.concatenate([self.fea_up(x_bridge), def_x_enc4], dim=1))  # [B,4*ch_num,Nt,Ny/8,Nx/8]
        vel = vel + self.vel_con(self.reg_head_dec4(x_dec4))  # [B,2,Nt,Ny/8,Nx/8]

        vel = self.vel_up(vel)  # [B,2,Nt,Ny/4,Nx/4]
        disp = self.vel_int(vel)  # [B,2,Nt,Ny/4,Nx/4]
        def_x_enc3 = self.spatial_transformer(x_enc3, disp)  # [B,4*ch_num,Nt,Ny/4,Nx/4]

        # Step 3
        x_dec3 = self.dec_block3(
            torch.concatenate([self.fea_up(x_dec4), def_x_enc3], dim=1))  # [B,2*ch_num,Nt,Ny/4,Nx/4]
        vel = vel + self.vel_con(self.reg_head_dec3(x_dec3))  # [B,2,Nt,Ny/4,Nx/4]

        vel = self.vel_up(vel)  # [B,2,Nt,Ny/2,Nx/2]
        disp = self.vel_int(vel)  # [B,2,Nt,Ny/2,Nx/2]
        def_x_enc2 = self.spatial_transformer(x_enc2, disp)  # [B,2*ch_num,Nt,Ny/2,Nx/2]

        # Step 4
        x_dec2 = self.dec_block2(torch.concatenate([self.fea_up(x_dec3), def_x_enc2], dim=1))  # [B,ch_num,Nt,Ny/2,Nx/2]
        vel = vel + self.vel_con(self.reg_head_dec2(x_dec2))  # [B,2,Nt,Ny/2,Nx/2]

        vel = self.vel_up(vel)  # [B,2,Nt,Ny,Nx]
        disp = self.vel_int(vel)  # [B,2,Nt,Ny,Nx]
        def_x_enc1 = self.spatial_transformer(x_enc1, disp)  # [B,ch_num,Nt,Ny,Nx]

        # Step 5
        x_dec1 = self.dec_block1(torch.concatenate([self.fea_up(x_dec2), def_x_enc1], dim=1))  # [B,ch_num,Nt,Ny,Nx]
        vel = vel + self.vel_con(self.reg_head_dec1(x_dec1))  # [B,2,Nt,Ny,Nx]

        disp = self.vel_int(vel)  # [B,2,Nt,Ny,Nx]
        inv_disp = self.vel_int(-vel)  # [B,2,Nt,Ny,Nx]
        def_x = self.spatial_transformer(x, disp)  # [B,1,Nt,Ny,Nx]

        # Reformat the deformed images and displacement fields
        def_x = torch.squeeze(def_x, dim=1)  # [B,Nt,Ny,Nx]
        tmp_vel = []
        tmp_disp = []
        tmp_inv_disp = []
        for t in range(vel.shape[2]):
            tmp_vel.append(vel[:, 0, t:t + 1, :, :])
            tmp_vel.append(vel[:, 1, t:t + 1, :, :])
            tmp_disp.append(disp[:, 0, t:t + 1, :, :])
            tmp_disp.append(disp[:, 1, t:t + 1, :, :])
            tmp_inv_disp.append(inv_disp[:, 0, t:t + 1, :, :])
            tmp_inv_disp.append(inv_disp[:, 1, t:t + 1, :, :])
        vel = torch.concatenate(tmp_vel, dim=1)  # [B,2*Nt,Ny,Nx]
        disp = torch.concatenate(tmp_disp, dim=1)  # [B,2*Nt,Ny,Nx]
        inv_disp = torch.concatenate(tmp_inv_disp, dim=1)  # [B,2*Nt,Ny,Nx]

        return def_x, vel, disp, inv_disp
