# %%
# Extract the pretrained .rar archive in place before running this script.
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from Models import Res_MLP

# Select the visible GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('Let\'s use', torch.cuda.device_count(), 'GPU!')
if not torch.cuda.is_available():
    raise RuntimeError('JMES requires a CUDA-enabled GPU.')
device = torch.device('cuda:0')

# Define the network
sig_dim_num = 11
TI_dim_num = 11
para_dim_num = 1
input_dim_num = sig_dim_num + TI_dim_num
output_dim_num = para_dim_num
width = 200
pe_net = Res_MLP(input_dim_num, output_dim_num, width)  # Parameter estimation network
pe_net = nn.DataParallel(pe_net)
pe_net.to(device)

# Create the testing dataset
time_scaling = 1000
batch_size = 1024

test_data_path = f'./Data/Simulation/Simulated_Signals/test.mat'
test_data = loadmat(test_data_path)
test_sig_record = test_data['sig_record']
test_TI_record = test_data['TI_record'] / time_scaling
test_para_record = test_data['para_record'] / time_scaling
test_sig_record, test_TI_record, test_para_record \
    = map(lambda x: torch.tensor(x, dtype=torch.float32), (test_sig_record, test_TI_record, test_para_record))
test_X = torch.concatenate((test_sig_record, test_TI_record), dim=1)
test_Y = test_para_record
test_ds = TensorDataset(test_X, test_Y)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# %% Perform testing
if __name__ == '__main__':
    para_load_path = f'./Learned_Parameters/PE_Net/pe_net.pth'
    pe_net.load_state_dict(torch.load(para_load_path, map_location=device, weights_only=True))
    pe_net.eval()
    test_xb_pred_record = torch.zeros(0, output_dim_num).to(device)
    test_yb_record = torch.zeros(0, output_dim_num).to(device)
    with torch.no_grad():
        for test_xb, test_yb in test_dl:
            test_xb, test_yb = test_xb.to(device), test_yb.to(device)

            test_xb_pred = pe_net(test_xb)

            test_xb_pred_record = torch.concatenate((test_xb_pred_record, test_xb_pred), dim=0)
            test_yb_record = torch.concatenate((test_yb_record, test_yb), dim=0)

        test_t1_error_record = torch.abs((test_xb_pred_record - test_yb_record) / (test_yb_record + 1e-5))
        test_t1_error_mean = torch.mean(test_t1_error_record)
        test_t1_error_std = torch.std(test_t1_error_record)

    print(f'T1 test error (%) = {100 * test_t1_error_mean:.1f} ({100 * test_t1_error_std:.1f})')

# %% Show some in vivo examples: estimated maps
if __name__ == '__main__':
    sub_ind = 1
    slice_ind = 1
    in_vivo_data_path = f'./Data/In_Vivo_Data/In_Vivo_Data_Preprocessed/' \
                        f'subject{sub_ind:d}/slice{slice_ind:d}/data.mat'
    in_vivo_data = loadmat(in_vivo_data_path)
    im = in_vivo_data['im']
    TI = in_vivo_data['TI'] / time_scaling
    im, TI = map(lambda x: torch.tensor(x, dtype=torch.float32), [im, TI])
    Ny, Nx = im.shape[0], im.shape[1]
    im = im.reshape([Ny * Nx, sig_dim_num])
    im = im / (torch.sqrt(torch.sum(im ** 2, dim=1)).reshape([Ny * Nx, 1]) + 1e-5)  # Normalization
    TI = torch.tile(TI.reshape([1, TI_dim_num]), [Ny * Nx, 1])
    im_TI = torch.concatenate((im, TI), dim=1)
    im_TI = im_TI.to(device)
    with torch.no_grad():
        pred = pe_net(im_TI)
    pred = pred.cpu()
    pred = torch.clamp(pred, min=50 / 1000, max=3000 / 1000)
    pred_T1_map = pred.reshape([Ny, Nx, output_dim_num]).numpy()
    pred_T1_map = pred_T1_map[:, :, 0] * time_scaling

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    plt.subplots_adjust(bottom=0.01, top=0.95, left=0.01, right=0.95)

    tmp = ax.imshow(pred_T1_map, cmap='jet', vmin=0, vmax=2200)
    fig.colorbar(tmp, ax=ax, shrink=0.9, pad=0.05)
    ax.axis('off')
    ax.set_title('pe_net, T1')

    fig.show()

    # Synthetic images
    im = in_vivo_data['im']

    pred_sig_map = torch.abs(1 - 2 * torch.exp(-TI / pred))
    pred_sig_map = pred_sig_map.reshape([Ny, Nx, sig_dim_num]).numpy()
    scale_map = (np.sum(im * pred_sig_map, axis=2, keepdims=True) /
                 (np.sum(pred_sig_map ** 2, axis=2, keepdims=True) + 1e-5))
    pred_sig_map = scale_map * pred_sig_map

    # Network-based vs. acquired
    fig, axs = plt.subplots(3, 11, figsize=(22, 6))
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

    save_path = f'./Results/PE/Maps_and_Synthetic_Images/subject{sub_ind:d}_slice{slice_ind:d}.mat'
    save_dict = {'im': im, 'pred_T1_map': pred_T1_map, 'pred_sig_map': pred_sig_map}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    savemat(save_path, save_dict)
