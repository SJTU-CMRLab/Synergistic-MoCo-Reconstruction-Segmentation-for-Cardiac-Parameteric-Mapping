# %%
# Extract the pretrained PE .rar archive in place before running this script.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
from Models import *

torch.set_num_threads(8)

# Select the visible GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('Let\'s use', torch.cuda.device_count(), 'GPU!')
if not torch.cuda.is_available():
    raise RuntimeError('JMES requires a CUDA-enabled GPU.')
device = torch.device('cuda:0')

# torch.autograd.set_detect_anomaly(True)  # For debugging

# Define the segmentation network
im_shape = [160, 160]
im_num = 12
class_num = 3
ch_num = 32
drop_rate = 0.1
seg_net = SegTransUNet2d(im_shape, im_num, class_num, ch_num, drop_rate)
seg_net = nn.DataParallel(seg_net).to(device)

# Define the registration network
frame_num = 11
reg_net = CDGRNet(im_shape, frame_num, ch_num, drop_rate)
reg_net = nn.DataParallel(reg_net).to(device)

# Define the spatial transformer
spatial_transformer = SpatialTransformer(im_shape)

# Load the pretrained parameter estimation network
sig_dim_num = 11  # Equates to frame_num
TI_dim_num = 11
para_dim_num = 1
pe_net_input_dim_num = sig_dim_num + TI_dim_num
pe_net_output_dim_num = para_dim_num
pe_net_width = 200
pe_net = Res_MLP(pe_net_input_dim_num, pe_net_output_dim_num, pe_net_width)
pe_net = nn.DataParallel(pe_net).to(device)
pe_net_para_load_path = f'./Learned_Parameters/PE_Net/pe_net.pth'
pe_net.load_state_dict(torch.load(pe_net_para_load_path, map_location=device, weights_only=True))
for para in pe_net.parameters():
    para.requires_grad = False

# Define the image synthesis block
image_synthesis = ImageSynthesis(pe_net)

# Define the loss function
dissimilarity = LNCC()  # LNCC dissimilarity for a pair of images
lr_constraint = LR()  # LR constraint for a sequence of images
regularizer = SpatialRegularizer(order=2)  # First or second order regularizer for a velocity field
seg_overlap = GDS()  # Generalized Dice overlap for a pair of segmentation masks
alpha = 0.04  # LR loss coefficient
beta = 60  # Regularizer coefficient
gamma = 4  # Segmentation loss coefficient

lf_num = 11  # Labeled frame number

# Define the optimizer
reg_net_learning_rate = 1e-4
seg_net_learning_rate = 1e-4
optimizer = torch.optim.Adam([{'params': reg_net.parameters(), 'lr': reg_net_learning_rate, 'weight_decay': 0},
                              {'params': seg_net.parameters(), 'lr': seg_net_learning_rate, 'weight_decay': 0}])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Learning rate decay

# Create the datasets
data_dir = f'./Data/In_Vivo_Data/In_Vivo_Data_Preprocessed'
slice_ind_record = [1, 2, 3, 4, 5]
batch_size = 4
# Training dataset
bilinear_mode = transforms.InterpolationMode.BILINEAR
bicubic_mode = transforms.InterpolationMode.BICUBIC
train_transforms1 = transforms.Compose([transforms.RandomAffine(degrees=(-30.0, 30.0), translate=(0.06, 0.06),
                                                                scale=(0.8, 1.2), shear=(0, 0, 0, 0),
                                                                interpolation=bilinear_mode, fill=0.0),
                                        transforms.ElasticTransform(alpha=600.0, sigma=25.0, interpolation=bicubic_mode,
                                                                    fill=0.0)])
train_transforms2 = transforms.Compose([transforms.RandomAffine(degrees=(0.0, 0.0), translate=(0.06, 0.12),
                                                                scale=(0.9, 1.1), shear=(0, 0, 0, 0),
                                                                interpolation=bilinear_mode, fill=0.0),
                                        transforms.ElasticTransform(alpha=600.0, sigma=25.0, interpolation=bicubic_mode,
                                                                    fill=0.0)])
train_ds = RegSegDataset(data_dir, train_sub_ind_record, slice_ind_record, im_shape=im_shape,
                         ds_transforms1=train_transforms1, ds_transforms2=train_transforms2)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# Validation dataset
valid_ds = RegSegDataset(data_dir, valid_sub_ind_record, slice_ind_record, im_shape=im_shape,
                         ds_transforms1=None, ds_transforms2=None)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

# %% Perform training
if __name__ == '__main__':
    # Create an empty log directory
    log_dir = './Runs/Reg_Seg_Nets'
    # shutil.rmtree(log_dir)
    # os.mkdir(log_dir)
    writer = SummaryWriter(log_dir, flush_secs=100)

    para_save_path = (f'./Learned_Parameters/Reg_Seg_Nets/CDGRNet_SegTransUNet2d_'
                      f'alpha={alpha:.2f}_beta={beta:.2f}_gamma={gamma:.2f}_drop={drop_rate:.2f}_lf_num={lf_num:d}.pth')

    epoch_num = 500
    iter_ind = 0
    min_valid_loss = torch.inf
    reg_net.train()
    seg_net.train()
    for epoch_ind in range(epoch_num):

        for train_xb_im, train_xb_TI, train_xb_true_mask, train_xb_frame_order in train_dl:

            train_xb_im, train_xb_TI, train_xb_true_mask = (
                train_xb_im.to(device), train_xb_TI.to(device), train_xb_true_mask.to(device))

            # Registration
            train_xb_def_im, train_xb_vel, _, train_xb_inv_disp = reg_net(train_xb_im)

            # Image synthesis
            train_xb_map, train_xb_syn_im = image_synthesis(train_xb_def_im, train_xb_TI, 'eval')

            # Segmentation
            tmp = torch.concatenate([train_xb_map, train_xb_def_im], dim=1)
            train_xb_prob, _ = seg_net(tmp)

            train_xb_Ld = 0  # DM loss
            train_xb_Lr = 0  # Regularizer loss
            for t in range(frame_num):
                train_xb_Ld = train_xb_Ld + dissimilarity(train_xb_def_im[:, t:t + 1, :, :],
                                                          train_xb_syn_im[:, t:t + 1, :, :])
                train_xb_Lr = train_xb_Lr + regularizer(train_xb_vel[:, 2 * t:2 * t + 2, :, :])
            train_xb_Ld = train_xb_Ld / frame_num
            train_xb_Lr = train_xb_Lr / frame_num
            train_xb_Ls = 0  # Segmentation loss
            if lf_num > 0:
                slp_num = train_xb_im.shape[0]
                for spl_ind in range(slp_num):
                    train_x_prob = train_xb_prob[spl_ind:spl_ind + 1, :, :, :]
                    train_x_inv_disp = train_xb_inv_disp[spl_ind:spl_ind + 1, :, :, :]
                    train_x_true_mask = train_xb_true_mask[spl_ind:spl_ind + 1, :, :, :]
                    train_x_frame_order = train_xb_frame_order[spl_ind]
                    for lf_ind in range(lf_num):
                        t = int(train_x_frame_order[lf_ind])
                        tmp = spatial_transformer(train_x_prob, train_x_inv_disp[:, 2 * t:2 * t + 2, :, :])
                        train_xb_Ls = train_xb_Ls + seg_overlap(tmp, train_x_true_mask[:, 3 * t:3 * t + 3, :, :])
                train_xb_Ls = train_xb_Ls / (slp_num * lf_num)
            train_xb_Ll = lr_constraint(train_xb_def_im)  # LR loss
            train_xb_loss = train_xb_Ld + alpha * train_xb_Ll + beta * train_xb_Lr + gamma * train_xb_Ls

            train_xb_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iter_ind % 100 == 0:
                reg_net.eval()
                seg_net.eval()
                with torch.no_grad():
                    valid_xb_Ld_record = []
                    valid_xb_Ll_record = []
                    valid_xb_Lr_record = []
                    valid_xb_Ls_record = []
                    for valid_xb_im, valid_xb_TI, valid_xb_true_mask, valid_xb_frame_order in valid_dl:

                        valid_xb_im, valid_xb_TI, valid_xb_true_mask = (
                            valid_xb_im.to(device), valid_xb_TI.to(device), valid_xb_true_mask.to(device))

                        # Registration
                        valid_xb_def_im, valid_xb_vel, _, valid_xb_inv_disp = reg_net(valid_xb_im)

                        # Image synthesis
                        valid_xb_map, valid_xb_syn_im = image_synthesis(valid_xb_def_im, valid_xb_TI, 'eval')

                        # Segmentation
                        tmp = torch.concatenate([valid_xb_map, valid_xb_def_im], dim=1)
                        valid_xb_prob, _ = seg_net(tmp)

                        valid_xb_Ld = 0  # DM loss
                        valid_xb_Lr = 0  # Regularizer loss
                        for t in range(frame_num):
                            valid_xb_Ld = valid_xb_Ld + dissimilarity(valid_xb_def_im[:, t:t + 1, :, :],
                                                                      valid_xb_syn_im[:, t:t + 1, :, :])
                            valid_xb_Lr = valid_xb_Lr + regularizer(valid_xb_vel[:, 2 * t:2 * t + 2, :, :])
                        valid_xb_Ld = valid_xb_Ld / frame_num
                        valid_xb_Lr = valid_xb_Lr / frame_num
                        valid_xb_Ls = 0  # Segmentation loss
                        if lf_num > 0:
                            slp_num = valid_xb_im.shape[0]
                            for spl_ind in range(slp_num):
                                valid_x_prob = valid_xb_prob[spl_ind:spl_ind + 1, :, :, :]
                                valid_x_inv_disp = valid_xb_inv_disp[spl_ind:spl_ind + 1, :, :, :]
                                valid_x_true_mask = valid_xb_true_mask[spl_ind:spl_ind + 1, :, :, :]
                                valid_x_frame_order = valid_xb_frame_order[spl_ind]
                                for lf_ind in range(lf_num):
                                    t = int(valid_x_frame_order[lf_ind])
                                    tmp = spatial_transformer(valid_x_prob, valid_x_inv_disp[:, 2 * t:2 * t + 2, :, :])
                                    valid_xb_Ls = valid_xb_Ls + seg_overlap(tmp, valid_x_true_mask[:, 3 * t:3 * t + 3, :, :])
                            valid_xb_Ls = valid_xb_Ls / (slp_num * lf_num)
                        valid_xb_Ll = lr_constraint(valid_xb_def_im)  # LR loss
                        valid_xb_Ld_record.append(valid_xb_Ld)
                        valid_xb_Ll_record.append(valid_xb_Ll)
                        valid_xb_Lr_record.append(valid_xb_Lr)
                        valid_xb_Ls_record.append(valid_xb_Ls)
                    valid_Ld = sum(valid_xb_Ld_record) / len(valid_xb_Ld_record)
                    valid_Ll = sum(valid_xb_Ll_record) / len(valid_xb_Ll_record)
                    valid_Lr = sum(valid_xb_Lr_record) / len(valid_xb_Lr_record)
                    valid_Ls = sum(valid_xb_Ls_record) / len(valid_xb_Ls_record)
                    valid_loss = valid_Ld + alpha * valid_Ll + beta * valid_Lr + gamma * valid_Ls
                reg_net.train()
                seg_net.train()

                if valid_loss < min_valid_loss:
                    para = {'reg_net': reg_net.state_dict(), 'seg_net': seg_net.state_dict()}
                    torch.save(para, para_save_path)
                    min_valid_loss = valid_loss
                    min_valid_Ld = valid_Ld
                    min_valid_Ll = valid_Ll
                    min_valid_Lr = valid_Lr
                    min_valid_Ls = valid_Ls
                    min_valid_loss_epoch_ind = epoch_ind
                    min_valid_loss_iter_ind = iter_ind

                writer.add_scalars('Train loss vs. Valid loss',
                                   {'Train': train_xb_loss, 'Valid': valid_loss}, iter_ind)

                print(f'Epoch {epoch_ind:d}, iter {iter_ind:d}: '
                      f'train loss = {train_xb_loss:.5f} ({train_xb_Ld:.5f}, {train_xb_Ll:.5f}, {train_xb_Lr:.5f}, '
                      f'{train_xb_Ls:.5f}), '
                      f'valid loss = {valid_loss:.5f} ({valid_Ld:.5f}, {valid_Ll:.5f}, {valid_Lr:.5f}, '
                      f'{valid_Ls:.5f}), '
                      f'learn rate = ({scheduler.get_last_lr()[0]:g}, {scheduler.get_last_lr()[1]:g})')

            iter_ind += 1

        if epoch_ind > 49:
            scheduler.step()

    print(f'\nEpoch {min_valid_loss_epoch_ind:d}, iter {min_valid_loss_iter_ind:d}: '
          f'minimal valid loss = {min_valid_loss:.5f} ({min_valid_Ld:.5f}, {min_valid_Ll:.5f}, {min_valid_Lr:.5f},'
          f' {min_valid_Ls:.5f})')
