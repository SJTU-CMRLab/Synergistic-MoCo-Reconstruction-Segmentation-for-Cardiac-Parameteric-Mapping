# %%
# Extract the pretrained .rar archives in place before running this script.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from Models import *

# Select the visible GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('Let\'s use', torch.cuda.device_count(), 'GPU!')
if not torch.cuda.is_available():
    raise RuntimeError('JMES requires a CUDA-enabled GPU.')
device = torch.device('cuda:0')

# Define the segmentation network
im_shape = [160, 160]
im_num = 12
class_num = 3
ch_num = 32
drop_rate = 0.1
seg_net = SegTransUNet2d(im_shape, im_num, class_num, ch_num, drop_rate)
seg_net = nn.DataParallel(seg_net).to(device)

# Define the registration network
frame_num = 10
reg_net = CDGRNet(im_shape, frame_num, ch_num, drop_rate)
reg_net = nn.DataParallel(reg_net).to(device)

# Define the spatial transformer
spatial_transformer = SpatialTransformer(im_shape)

# Load the pretrained parameter estimation network
sig_dim_num = 10  # Equates to frame_num
RR_dim_num = 9
para_dim_num = 2
pe_net_input_dim_num = sig_dim_num + RR_dim_num
pe_net_output_dim_num = para_dim_num
pe_net_width = 200
pe_net = Res_MLP(pe_net_input_dim_num, pe_net_output_dim_num, pe_net_width)
pe_net = nn.DataParallel(pe_net).to(device)
pe_net_para_load_path = f'./Learned_Parameters/PE_Net/pe_net.pth'
pe_net.load_state_dict(torch.load(pe_net_para_load_path, map_location=device, weights_only=True))
for para in pe_net.parameters():
    para.requires_grad = False

# Load the pretrained signal simulation network
ss_net_input_dim_num = para_dim_num + RR_dim_num
ss_net_output_dim_num = sig_dim_num
ss_net_width = 200
ss_net = MLP(ss_net_input_dim_num, ss_net_output_dim_num, ss_net_width)
ss_net = nn.DataParallel(ss_net).to(device)
ss_net_para_load_path = f'./Learned_Parameters/SS_Net/ss_net.pth'
ss_net.load_state_dict(torch.load(ss_net_para_load_path, map_location=device, weights_only=True))
for para in ss_net.parameters():
    para.requires_grad = False

# Define the image synthesis block
image_synthesis = ImageSynthesis(pe_net, ss_net)

# Define the parameter estimation block
parameter_estimation = ParameterEstimation(pe_net)

# Define the loss function
dissimilarity = LNCC()  # LNCC dissimilarity for a pair of images
lr_constraint = LR()  # LR constraint for a sequence of images
regularizer = SpatialRegularizer(order=2)  # First or second order regularizer for a velocity field
seg_overlap = GDS()  # Generalized Dice overlap for a pair of segmentation masks
alpha = 0.04  # LR loss coefficient
beta = 60  # Regularizer coefficient
gamma = 4  # Segmentation loss coefficient

lf_num = 10  # Labeled frame number

# Create the testing dataset
data_dir = f'./Data/In_Vivo_Data/In_Vivo_Data_Preprocessed'
slice_ind_record = [1, 2, 3]
batch_size = 4
# Public examples bundled with this repository; this is not the test cohort used in our study.
test_sub_ind_record = [1, 2, 3, 4, 5]
test_ds = RegSegDataset(data_dir, test_sub_ind_record, slice_ind_record, im_shape=im_shape,
                        ds_transforms1=None, ds_transforms2=None)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

para_num = sum([para.nelement() for para in reg_net.parameters()])
print(f'Number of the reg_net parameters: {para_num / 1048576:.1f}M')

# %% Perform testing
if __name__ == '__main__':
    para_load_path = (f'./Learned_Parameters/Reg_Seg_Nets/CDGRNet_SegTransUNet2d_'
                      f'alpha={alpha:.2f}_beta={beta:.2f}_gamma={gamma:.2f}_drop={drop_rate:.2f}_lf_num={lf_num:d}.pth')
    para = torch.load(para_load_path, map_location=device, weights_only=True)
    reg_net.load_state_dict(para['reg_net'])
    seg_net.load_state_dict(para['seg_net'])
    reg_net.eval()
    seg_net.eval()
    with torch.no_grad():
        test_xb_Ld_record = []
        test_xb_Ll_record = []
        test_xb_Lr_record = []
        test_xb_Ls_record = []
        for test_xb_im, test_xb_RR, test_xb_true_mask, _ in test_dl:

            test_xb_im, test_xb_RR, test_xb_true_mask = (
                test_xb_im.to(device), test_xb_RR.to(device), test_xb_true_mask.to(device))

            # Registration
            test_xb_def_im, test_xb_vel, _, test_xb_inv_disp = reg_net(test_xb_im)

            # Image synthesis
            test_xb_map, test_xb_syn_im = image_synthesis(test_xb_def_im, test_xb_RR, 'eval')

            # Segmentation
            tmp = torch.concatenate([test_xb_map, test_xb_def_im], dim=1)
            test_xb_prob, _ = seg_net(tmp)

            test_xb_Ld = 0  # DM loss
            test_xb_Lr = 0  # Regularizer loss
            test_xb_Ls = 0  # Segmentation loss
            for t in range(frame_num):
                test_xb_Ld = test_xb_Ld + dissimilarity(test_xb_def_im[:, t:t + 1, :, :],
                                                        test_xb_syn_im[:, t:t + 1, :, :])
                test_xb_Lr = test_xb_Lr + regularizer(test_xb_vel[:, 2 * t:2 * t + 2, :, :])
                tmp = spatial_transformer(test_xb_prob, test_xb_inv_disp[:, 2 * t:2 * t + 2, :, :])
                test_xb_Ls = test_xb_Ls + seg_overlap(tmp, test_xb_true_mask[:, 3 * t:3 * t + 3, :, :])
            test_xb_Ld = test_xb_Ld / frame_num
            test_xb_Lr = test_xb_Lr / frame_num
            test_xb_Ls = test_xb_Ls / frame_num
            test_xb_Ll = lr_constraint(test_xb_def_im)  # LR loss
            test_xb_Ld_record.append(test_xb_Ld)
            test_xb_Ll_record.append(test_xb_Ll)
            test_xb_Lr_record.append(test_xb_Lr)
            test_xb_Ls_record.append(test_xb_Ls)
        test_Ld = sum(test_xb_Ld_record) / len(test_xb_Ld_record)
        test_Ll = sum(test_xb_Ll_record) / len(test_xb_Ll_record)
        test_Lr = sum(test_xb_Lr_record) / len(test_xb_Lr_record)
        test_Ls = sum(test_xb_Ls_record) / len(test_xb_Ls_record)
        test_loss = test_Ld + alpha * test_Ll + beta * test_Lr + gamma * test_Ls

    print(f'Test loss = {test_loss:.5f} ({test_Ld:.5f}, {test_Ll:.5f}, {test_Lr:.5f}, {test_Ls:.5f})')

# %% Show some in vivo examples: motion-corrected images and maps
if __name__ == '__main__':
    sub_ind = 1
    slice_ind = 1
    im, RR = RegDataset(data_dir, [sub_ind], [slice_ind], ds_transforms=transforms.CenterCrop(size=im_shape))[0]
    im, RR = map(lambda x: torch.unsqueeze(x, dim=0).to(device), [im, RR])

    time_scaling = 1000
    with torch.no_grad():
        pmap = parameter_estimation(im, RR) * time_scaling

        new_im, _, disp, inv_disp = reg_net(im)
        new_pmap = parameter_estimation(new_im, RR)
        tmp = torch.concatenate([new_pmap, new_im], dim=1)
        prob, pred_mask = seg_net(tmp)
        new_pmap = new_pmap * time_scaling

    one_im = torch.ones(im.shape[2:4])
    alpha_mask = (pred_mask[0, 0, :, :].cpu() == 1).to(torch.float32)

    # Motion-corrected images
    fig, axs = plt.subplots(2, 10, figsize=(20, 4))
    plt.subplots_adjust(bottom=0.005, top=0.95, left=0.005, right=0.995)

    for i in range(frame_num):
        axs[0][i].imshow(im[0, i, :, :].cpu(), cmap='gray', vmin=0, vmax=0.75)
        axs[0][i].axis('off')
        axs[0][i].set_title(f'original, {i:d}')

        axs[1][i].imshow(new_im[0, i, :, :].cpu(), cmap='gray', vmin=0, vmax=0.75)
        axs[1][i].axis('off')
        axs[1][i].set_title(f'corrected')

    plt.show()

    # Segmented motion-corrected maps
    fig, axs = plt.subplots(2, 4, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.01, top=0.95, left=0.01, right=0.99)

    tmp = axs[0][0].imshow(pmap[0, 0, :, :].cpu(), cmap='jet', vmin=0, vmax=2200)
    fig.colorbar(tmp, ax=axs[0][0], shrink=0.9, pad=0.05)
    axs[0][0].axis('off')
    axs[0][0].set_title('original T1')

    tmp = axs[0][1].imshow(new_pmap[0, 0, :, :].cpu(), cmap='jet', vmin=0, vmax=2200)
    fig.colorbar(tmp, ax=axs[0][1], shrink=0.9, pad=0.05)
    axs[0][1].axis('off')
    axs[0][1].set_title('corrected T1')

    tmp = axs[0][2].imshow(new_pmap[0, 0, :, :].cpu(), cmap='jet', vmin=0, vmax=2200)
    fig.colorbar(tmp, ax=axs[0][2], shrink=0.9, pad=0.05)
    axs[0][2].imshow(one_im, cmap='Blues', vmin=0, vmax=1, alpha=alpha_mask)
    axs[0][2].axis('off')
    axs[0][2].set_title('segmented T1')

    tmp = axs[0][3].imshow(pred_mask[0, 0, :, :].cpu(), cmap='gray', vmin=0, vmax=2)
    fig.colorbar(tmp, ax=axs[0][3], shrink=0.9, pad=0.05)
    axs[0][3].axis('off')
    axs[0][3].set_title('mask')

    tmp = axs[1][0].imshow(pmap[0, 1, :, :].cpu(), cmap='viridis', vmin=0, vmax=170)
    fig.colorbar(tmp, ax=axs[1][0], shrink=0.9, pad=0.05)
    axs[1][0].axis('off')
    axs[1][0].set_title('original T2')

    tmp = axs[1][1].imshow(new_pmap[0, 1, :, :].cpu(), cmap='viridis', vmin=0, vmax=170)
    fig.colorbar(tmp, ax=axs[1][1], shrink=0.9, pad=0.05)
    axs[1][1].axis('off')
    axs[1][1].set_title('corrected T2')

    tmp = axs[1][2].imshow(new_pmap[0, 1, :, :].cpu(), cmap='viridis', vmin=0, vmax=170)
    fig.colorbar(tmp, ax=axs[1][2], shrink=0.9, pad=0.05)
    axs[1][2].imshow(one_im, cmap='Reds', vmin=0, vmax=1, alpha=alpha_mask)
    axs[1][2].axis('off')
    axs[1][2].set_title('segmented T2')

    tmp = axs[1][3].imshow(prob[0, 1, :, :].cpu(), cmap='gray', vmin=0, vmax=1)
    fig.colorbar(tmp, ax=axs[1][3], shrink=0.9, pad=0.05)
    axs[1][3].axis('off')
    axs[1][3].set_title('myo prob')

    plt.show()

# %% Process the specified samples and save the results
if __name__ == '__main__':
    sub_ind_record = test_sub_ind_record  # Testing dataset
    slice_ind_record = list(range(1, 4))

    run_time_list = []
    for sub_ind in sub_ind_record:
        for slice_ind in slice_ind_record:

            im, RR, true_mask, _ = RegSegDataset(data_dir, [sub_ind], [slice_ind], im_shape=im_shape,
                                                 ds_transforms1=None, ds_transforms2=None)[0]
            im, RR, true_mask = map(lambda x: torch.unsqueeze(x, dim=0).to(device), [im, RR, true_mask])

            time_scaling = 1000
            with torch.no_grad():
                true_mask = torch.argmax(
                    true_mask.reshape(im.shape[0], frame_num, 3, im.shape[2], im.shape[3]), dim=2)

                pmap, syn_im = image_synthesis(im, RR, 'eval')
                pmap = pmap * time_scaling

                # Process and record the run time
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                new_im, vel, disp, inv_disp = reg_net(im)
                new_pmap, new_syn_im = image_synthesis(new_im, RR, 'eval')
                tmp = torch.concatenate([new_pmap, new_im], dim=1)
                _, pred_mask = seg_net(tmp)
                end.record()
                torch.cuda.synchronize()  # Waits for everything to finish running
                run_time = start.elapsed_time(end) / 1000
                run_time_list.append(run_time)

                RR = RR * time_scaling
                new_pmap = new_pmap * time_scaling

            save_path = f'./Results/Reg_Seg/Corrected_Images_and_Maps/subject{sub_ind:d}_slice{slice_ind:d}.mat'
            RR = RR[0].cpu().numpy()
            im = torch.permute(im[0], [1, 2, 0]).cpu().numpy()
            new_im = torch.permute(new_im[0], [1, 2, 0]).cpu().numpy()
            pmap = pmap[0, :, :, :].cpu().numpy()
            new_pmap = new_pmap[0, :, :, :].cpu().numpy()
            syn_im = torch.permute(syn_im[0], [1, 2, 0]).cpu().numpy()
            new_syn_im = torch.permute(new_syn_im[0], [1, 2, 0]).cpu().numpy()
            vel = torch.permute(vel[0], [1, 2, 0]).cpu().numpy()
            disp = torch.permute(disp[0], [1, 2, 0]).cpu().numpy()
            inv_disp = torch.permute(inv_disp[0], [1, 2, 0]).cpu().numpy()
            pred_mask = pred_mask[0, 0, :, :].cpu().numpy()
            true_mask = torch.permute(true_mask[0], [1, 2, 0]).cpu().numpy()
            save_dict = {'RR': RR,
                         'im': im, 'new_im': new_im,
                         'T1_map': pmap[0, :, :], 'new_T1_map': new_pmap[0, :, :],
                         'T2_map': pmap[1, :, :], 'new_T2_map': new_pmap[1, :, :],
                         'syn_im': syn_im, 'new_syn_im': new_syn_im,
                         'vel_field_x': vel[:, :, 0::2], 'vel_field_y': vel[:, :, 1::2],
                         'disp_field_x': disp[:, :, 0::2], 'disp_field_y': disp[:, :, 1::2],
                         'inv_disp_field_x': inv_disp[:, :, 0::2], 'inv_disp_field_y': inv_disp[:, :, 1::2],
                         'pred_mask': pred_mask, 'true_mask': true_mask, 'run_time': run_time}
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            savemat(save_path, save_dict)

    print(f'\nRun time (s): {np.mean(run_time_list):.5f} ({np.std(run_time_list, ddof=1):.5f})\n')
