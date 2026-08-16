# %%
# Extract the pretrained .rar archive in place before running this script.
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Models import Res_MLP

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
input_dim_num = sig_dim_num + RR_dim_num
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
test_RR_record = test_data['RR_record'] / time_scaling
test_para_record = test_data['para_record'][:, 0:2] / time_scaling
test_sig_record, test_RR_record, test_para_record \
    = map(lambda x: torch.tensor(x, dtype=torch.float32), (test_sig_record, test_RR_record, test_para_record))
test_X = torch.concatenate((test_sig_record, test_RR_record), dim=1)
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

        test_t1_error_record = torch.abs((test_xb_pred_record[:, 0:1] - test_yb_record[:, 0:1])
                                         / (test_yb_record[:, 0:1] + 1e-5))
        test_t1_error_mean = torch.mean(test_t1_error_record)
        test_t1_error_std = torch.std(test_t1_error_record)
        test_t2_error_record = torch.abs((test_xb_pred_record[:, 1:2] - test_yb_record[:, 1:2])
                                         / (test_yb_record[:, 1:2] + 1e-5))
        test_t2_error_mean = torch.mean(test_t2_error_record)
        test_t2_error_std = torch.std(test_t2_error_record)

    print(f'T1 test error (%) = {100 * test_t1_error_mean:.1f} ({100 * test_t1_error_std:.1f})\n'
          f'T2 test error (%) = {100 * test_t2_error_mean:.1f} ({100 * test_t2_error_std:.1f})')

# %% Show some in vivo examples: estimated maps
if __name__ == '__main__':
    sub_ind = 1
    slice_ind = 1
    in_vivo_data_path = f'./Data/In_Vivo_Data/In_Vivo_Data_Preprocessed/' \
                        f'subject{sub_ind:d}/slice{slice_ind:d}/data.mat'
    in_vivo_data = loadmat(in_vivo_data_path)
    im = in_vivo_data['im']
    RR = in_vivo_data['RR'] / time_scaling
    im, RR = map(lambda x: torch.tensor(x, dtype=torch.float32), (im, RR))
    Ny, Nx = im.shape[0], im.shape[1]
    im = im.reshape([Ny * Nx, sig_dim_num])
    im = im / torch.sqrt(torch.sum(im ** 2, dim=1)).reshape([Ny * Nx, 1])  # Normalization
    RR = torch.tile(RR.reshape([1, RR_dim_num]), [Ny * Nx, 1])
    im_RR = torch.concatenate((im, RR), dim=1)
    im_RR = im_RR.to(device)
    with torch.no_grad():
        pred = pe_net(im_RR)
    pred = pred.reshape([Ny, Nx, output_dim_num]).cpu().numpy()
    pred_T1_map = pred[:, :, 0] * time_scaling
    pred_T2_map = pred[:, :, 1] * time_scaling

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

    fig.show()
