# %%
# Extract the pretrained .rar archives in place before running this script.
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from Models import Res_MLP, MLP

# Select the visible GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('Let\'s use', torch.cuda.device_count(), 'GPU!')
if not torch.cuda.is_available():
    raise RuntimeError('JMES requires a CUDA-enabled GPU.')
device = torch.device('cuda:0')

# Define the network
sig_dim_num = 10
RR_dim_num = 9
para_dim_num = 2
ss_net_input_dim_num = para_dim_num + RR_dim_num
ss_net_output_dim_num = sig_dim_num
ss_net_width = 200
ss_net = MLP(ss_net_input_dim_num, ss_net_output_dim_num, ss_net_width)  # Signal simulation network
ss_net = nn.DataParallel(ss_net)
ss_net.to(device)

# Create the testing dataset
time_scaling = 1000
batch_size = 1024

test_data_path = f'./Data/Simulation/Simulated_Signals/test.mat'
test_data = loadmat(test_data_path)
test_sig_record = test_data['sig_record']
test_RR_record = test_data['RR_record'] / time_scaling
test_para_record = test_data['para_record'][:, 0:2] / time_scaling
test_sig_record, test_RR_record, test_para_record \
    = map(lambda x: torch.tensor(x, dtype=torch.float32), (test_sig_record, test_RR_record, test_para_record))
test_X = torch.concatenate((test_para_record, test_RR_record), dim=1)
test_Y = test_sig_record
test_ds = TensorDataset(test_X, test_Y)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# %% Perform testing
if __name__ == '__main__':
    para_load_path = f'./Learned_Parameters/SS_Net/ss_net.pth'
    ss_net.load_state_dict(torch.load(para_load_path, map_location=device, weights_only=True))
    ss_net.eval()
    test_xb_pred_record = torch.zeros(0, ss_net_output_dim_num).to(device)
    test_yb_record = torch.zeros(0, ss_net_output_dim_num).to(device)
    with torch.no_grad():
        for test_xb, test_yb in test_dl:
            test_xb, test_yb = test_xb.to(device), test_yb.to(device)

            test_xb_pred = ss_net(test_xb)

            test_xb_pred_record = torch.concatenate((test_xb_pred_record, test_xb_pred), dim=0)
            test_yb_record = torch.concatenate((test_yb_record, test_yb), dim=0)

        test_error_record = torch.sqrt(torch.sum((test_xb_pred_record - test_yb_record) ** 2, dim=1)
                                       / (torch.sum(test_yb_record ** 2, dim=1) + 1e-5))
        test_error_mean = torch.mean(test_error_record)
        test_error_std = torch.std(test_error_record)

    print(f'Test error = {100 * test_error_mean:.1f} ({100 * test_error_std:.1f})')

# %% Show some examples in the testing dataset
if __name__ == '__main__':
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=True)
    test_xb, test_yb = next(iter(test_dl))
    with torch.no_grad():
        test_pred = ss_net(test_xb.to(device))
    test_pred = test_pred.cpu().numpy()
    test_xb, test_yb = test_xb.numpy(), test_yb.numpy()
    test_xb[:, 0:2] = test_xb[:, 0:2] * time_scaling

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    plt.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.99)

    t = np.arange(1, 11, 1)

    axs[0][0].plot(t, test_pred[0, :], label='SS')
    axs[0][0].plot(t, test_yb[0, :], label='EPG')
    axs[0][0].set_xlim((0.5, 10.5))
    axs[0][0].set_ylim((0, 0.6))
    axs[0][0].legend(loc='best')
    axs[0][0].set_title(f'T1={test_xb[0][0]:.0f}ms, T2={test_xb[0][1]:.0f}ms')

    axs[0][1].plot(t, test_pred[1, :], label='SS')
    axs[0][1].plot(t, test_yb[1, :], label='EPG')
    axs[0][1].set_xlim((0.5, 10.5))
    axs[0][1].set_ylim((0, 0.6))
    axs[0][1].legend(loc='best')
    axs[0][1].set_title(f'T1={test_xb[1][0]:.0f}ms, T2={test_xb[1][1]:.0f}ms')

    axs[1][0].plot(t, test_pred[2, :], label='SS')
    axs[1][0].plot(t, test_yb[2, :], label='EPG')
    axs[1][0].set_xlim((0.5, 10.5))
    axs[1][0].set_ylim((0, 0.6))
    axs[1][0].legend(loc='best')
    axs[1][0].set_title(f'T1={test_xb[2][0]:.0f}ms, T2={test_xb[2][1]:.0f}ms')

    axs[1][1].plot(t, test_pred[3, :], label='SS')
    axs[1][1].plot(t, test_yb[3, :], label='EPG')
    axs[1][1].set_xlim((0.5, 10.5))
    axs[1][1].set_ylim((0, 0.6))
    axs[1][1].legend(loc='best')
    axs[1][1].set_title(f'T1={test_xb[3][0]:.0f}ms, T2={test_xb[3][1]:.0f}ms')

    plt.show()

# %% Load the pretrained parameter estimation network
if __name__ == '__main__':
    pe_net_input_dim_num = sig_dim_num + RR_dim_num
    pe_net_output_dim_num = para_dim_num
    pe_net_width = 200
    pe_net = Res_MLP(pe_net_input_dim_num, pe_net_output_dim_num, pe_net_width)  # Parameter estimation network
    pe_net = nn.DataParallel(pe_net)
    pe_net.to(device)

    para_load_path = f'./Learned_Parameters/PE_Net/pe_net.pth'
    pe_net.load_state_dict(torch.load(para_load_path, map_location=device, weights_only=True))
    pe_net.eval()

# %% Show some in vivo examples: estimated maps and synthetic images
if __name__ == '__main__':
    sub_ind = 1
    slice_ind = 1
    in_vivo_data_path = f'./Data/In_Vivo_Data/In_Vivo_Data_Preprocessed/' \
                        f'subject{sub_ind:d}/slice{slice_ind:d}/data.mat'
    in_vivo_data = loadmat(in_vivo_data_path)
    # Estimated maps
    im = in_vivo_data['im']
    RR = in_vivo_data['RR'] / time_scaling
    im, RR = map(lambda x: torch.tensor(x, dtype=torch.float32), (im, RR))
    Ny, Nx = im.shape[0], im.shape[1]
    im = im.reshape([Ny * Nx, sig_dim_num])
    im = im / (torch.sqrt(torch.sum(im ** 2, dim=1)).reshape([Ny * Nx, 1]) + 1e-5)  # Normalization
    RR = torch.tile(RR.reshape([1, RR_dim_num]), [Ny * Nx, 1])
    im_RR = torch.concatenate((im, RR), dim=1)
    im_RR = im_RR.to(device)
    with torch.no_grad():
        pe_net_pred = pe_net(im_RR)
    pe_net_pred = pe_net_pred.cpu()
    pe_net_pred[:, 0] = torch.clamp(pe_net_pred[:, 0], min=50 / 1000, max=3000 / 1000)
    pe_net_pred[:, 1] = torch.clamp(pe_net_pred[:, 1], min=5 / 1000, max=250 / 1000)
    tmp = pe_net_pred.reshape([Ny, Nx, pe_net_output_dim_num]).numpy()
    pred_T1_map = tmp[:, :, 0] * time_scaling
    pred_T2_map = tmp[:, :, 1] * time_scaling

    fig, axs = plt.subplots(1, 2, figsize=(5, 2))
    plt.subplots_adjust(bottom=0.01, top=0.95, left=0.01, right=0.95)

    tmp = axs[0].imshow(pred_T1_map, cmap='jet', vmin=0, vmax=2200)
    fig.colorbar(tmp, ax=axs[0], shrink=0.9, pad=0.05)
    axs[0].axis('off')
    axs[0].set_title('pe_net, T1')

    tmp = axs[1].imshow(pred_T2_map, cmap='viridis', vmin=0, vmax=170)
    fig.colorbar(tmp, ax=axs[1], shrink=0.9, pad=0.05)
    axs[1].axis('off')
    axs[1].set_title('pe_net, T2')

    plt.show()

    # Synthetic images
    im = in_vivo_data['im']

    pe_net_pred_RR = torch.concatenate((pe_net_pred, RR), dim=1)
    pe_net_pred_RR = pe_net_pred_RR.to(device)
    with torch.no_grad():
        ss_net_pred = ss_net(pe_net_pred_RR)
    ss_net_pred = ss_net_pred.cpu()
    pred_sig_map = ss_net_pred.reshape([Ny, Nx, ss_net_output_dim_num]).numpy()
    scale_map = (np.sum(im * pred_sig_map, axis=2, keepdims=True) /
                 (np.sum(pred_sig_map ** 2, axis=2, keepdims=True) + 1e-5))
    pred_sig_map = scale_map * pred_sig_map

    # Network-based vs. acquired
    fig, axs = plt.subplots(3, 10, figsize=(20, 6))
    plt.subplots_adjust(bottom=0.005, top=0.95, left=0.005, right=0.995)

    for i in range(sig_dim_num):
        axs[0][i].imshow(pred_sig_map[:, :, i], cmap='gray', vmin=0, vmax=0.75)
        axs[0][i].axis('off')
        axs[0][i].set_title(f'net, {i:d}')

    for i in range(sig_dim_num):
        axs[1][i].imshow(im[:, :, i], cmap='gray', vmin=0, vmax=0.75)
        axs[1][i].axis('off')
        axs[1][i].set_title(f'acquired, {i:d}')

    for i in range(sig_dim_num):
        axs[2][i].imshow(np.abs(pred_sig_map[:, :, i] - im[:, :, i]), cmap='gray', vmin=0, vmax=0.05)
        axs[2][i].axis('off')
        axs[2][i].set_title(f'diff, {i:d}')

    plt.show()

    save_path = f'./Results/PE_and_SS/Maps_and_Synthetic_Images/' \
                f'subject{sub_ind:d}_slice{slice_ind:d}.mat'
    save_dict = {'im': im, 'pred_T1_map': pred_T1_map, 'pred_T2_map': pred_T2_map,
                 'pred_sig_map': pred_sig_map}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    savemat(save_path, save_dict)
