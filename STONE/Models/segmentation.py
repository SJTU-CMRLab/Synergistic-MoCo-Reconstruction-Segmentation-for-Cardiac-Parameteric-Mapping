import torch
import torch.nn as nn
from einops import rearrange


class SelfAttention(nn.Module):
    # Self-attention block without embedding
    def __init__(self, emb_dim, head_num, drop_rate):
        super().__init__()

        # Multi-head attention
        self.layer_norm_mha = nn.LayerNorm(emb_dim)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=head_num,
                                                          batch_first=True, dropout=drop_rate)
        self.dropout_mha = nn.Dropout(drop_rate)

        # Multi-layer perceptron (GEGLU)
        self.layer_norm_mlp = nn.LayerNorm(emb_dim)
        self.linear_mlp11 = nn.Linear(emb_dim, 2 * emb_dim)
        self.linear_mlp12 = nn.Linear(emb_dim, 2 * emb_dim)
        self.gelu = nn.GELU()
        self.dropout_mlp1 = nn.Dropout(drop_rate)
        self.linear_mlp2 = nn.Linear(2 * emb_dim, emb_dim)
        self.dropout_mlp2 = nn.Dropout(drop_rate)

    def forward(self, x):
        # x shape: [B,emb_dim,Ny,Nx]
        _, _, Ny, _ = x.shape

        # Sequence rearrangement
        x = rearrange(x, 'B C Ny Nx -> B (Ny Nx) C')  # [B,Ny*Nx,emb_dim]

        # Multi-head attention
        x_mha = self.layer_norm_mha(x)  # [B,Ny*Nx,emb_dim]
        x_mha, _ = self.multi_head_attention(x_mha, x_mha, x_mha, need_weights=False)  # [B,Ny*Nx,emb_dim]
        x_mha = self.dropout_mha(x_mha)  # [B,Ny*Nx,emb_dim]
        x_mha = x_mha + x  # [B,Ny*Nx,emb_dim]

        # Multi-layer perceptron (GEGLU)
        x_mlp = self.layer_norm_mlp(x_mha)  # [B,Ny*Nx,emb_dim]
        x_mlp = self.gelu(self.linear_mlp11(x_mlp)) * self.linear_mlp12(x_mlp)  # [B,Ny*Nx,2*emb_dim]
        x_mlp = self.dropout_mlp1(x_mlp)  # [B,Ny*Nx,2*emb_dim]
        x_mlp = self.linear_mlp2(x_mlp)  # [B,Ny*Nx,emb_dim]
        x_mlp = self.dropout_mlp2(x_mlp)  # [B,Ny*Nx,emb_dim]
        x_mlp = x_mlp + x_mha  # [B,Ny*Nx,emb_dim]

        # Image rearrangement
        x_mlp = rearrange(x_mlp, 'B (Ny Nx) C -> B C Ny Nx', Ny=Ny)  # [B,emb_dim,Ny,Nx]

        return x_mlp


def build_conv2d_block(in_ch_num, out_ch_num1, out_ch_num2):
    return nn.Sequential(
        nn.Conv2d(in_ch_num, out_ch_num1, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(num_groups=16, num_channels=out_ch_num1),
        nn.LeakyReLU(),
        nn.Conv2d(out_ch_num1, out_ch_num2, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(num_groups=16, num_channels=out_ch_num2),
        nn.LeakyReLU()
    )


class SegTransUNet2d(nn.Module):
    def __init__(self, im_shape, im_num, class_num, ch_num, drop_rate):
        # im_num: para_dim_num+Nt
        super().__init__()

        self.im_num = im_num

        # Layers
        self.enc_block1 = build_conv2d_block(im_num, ch_num, ch_num)
        self.enc_block2 = build_conv2d_block(ch_num, 2 * ch_num, 2 * ch_num)
        self.enc_block3 = build_conv2d_block(2 * ch_num, 4 * ch_num, 4 * ch_num)
        self.enc_block4 = build_conv2d_block(4 * ch_num, 8 * ch_num, 8 * ch_num)
        self.bridge_block = build_conv2d_block(8 * ch_num, 16 * ch_num, 16 * ch_num)

        self.patch_to_emb = nn.Conv2d(16 * ch_num, 8 * ch_num, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.pos_enc = nn.Parameter(torch.randn((1, 8 * ch_num, int(im_shape[0] / 16), int(im_shape[1] / 16))))
        self.sa_block1 = SelfAttention(emb_dim=8 * ch_num, head_num=int(ch_num / 2), drop_rate=drop_rate)
        self.sa_block2 = SelfAttention(emb_dim=8 * ch_num, head_num=int(ch_num / 2), drop_rate=drop_rate)
        self.sa_block3 = SelfAttention(emb_dim=8 * ch_num, head_num=int(ch_num / 2), drop_rate=drop_rate)
        self.sa_block4 = SelfAttention(emb_dim=8 * ch_num, head_num=int(ch_num / 2), drop_rate=drop_rate)
        self.sa_block5 = SelfAttention(emb_dim=8 * ch_num, head_num=int(ch_num / 2), drop_rate=drop_rate)
        self.sa_block6 = SelfAttention(emb_dim=8 * ch_num, head_num=int(ch_num / 2), drop_rate=drop_rate)

        self.dec_block4 = build_conv2d_block(16 * ch_num, 8 * ch_num, 4 * ch_num)
        self.dec_block3 = build_conv2d_block(8 * ch_num, 4 * ch_num, 2 * ch_num)
        self.dec_block2 = build_conv2d_block(4 * ch_num, 2 * ch_num, ch_num)
        self.dec_block1 = build_conv2d_block(2 * ch_num, ch_num, ch_num)
        self.output_block = nn.Conv2d(ch_num, class_num, kernel_size=3, stride=1, padding=1)

        self.down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_enc1 = self.enc_block1(x)
        x_enc2 = self.enc_block2(self.down(x_enc1))
        x_enc3 = self.enc_block3(self.down(x_enc2))
        x_enc4 = self.enc_block4(self.down(x_enc3))
        x_bridge = self.bridge_block(self.down(x_enc4))

        # Patch embedding
        x_bridge = self.patch_to_emb(x_bridge)

        # Positional encoding
        x_bridge = x_bridge + self.pos_enc

        # Self-attention
        x_bridge = self.sa_block1(x_bridge)
        x_bridge = self.sa_block2(x_bridge)
        x_bridge = self.sa_block3(x_bridge)
        x_bridge = self.sa_block4(x_bridge)
        x_bridge = self.sa_block5(x_bridge)
        x_bridge = self.sa_block6(x_bridge)

        x_dec4 = self.dec_block4(torch.concatenate([self.up(x_bridge), x_enc4], dim=1))
        x_dec3 = self.dec_block3(torch.concatenate([self.up(x_dec4), x_enc3], dim=1))
        x_dec2 = self.dec_block2(torch.concatenate([self.up(x_dec3), x_enc2], dim=1))
        x_dec1 = self.dec_block1(torch.concatenate([self.up(x_dec2), x_enc1], dim=1))
        prob = self.softmax(self.output_block(x_dec1))
        mask = torch.argmax(prob, dim=1, keepdim=True)

        return prob, mask
