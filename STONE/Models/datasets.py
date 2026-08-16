import torch
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
from torchvision import transforms

train_sub_ind_record = []
valid_sub_ind_record = []
test_sub_ind_record = []


class RegDataset(Dataset):
    # Dataset class for registration
    def __init__(self, data_dir, sub_ind_record, slice_ind_record, ds_transforms):
        time_scaling = 1000

        im_record = []
        TI_record = []
        for sub_ind in sub_ind_record:

            data_dir2 = os.path.join(data_dir, f'subject{sub_ind:d}')
            for slice_ind in slice_ind_record:
                data_path = os.path.join(data_dir2, f'slice{slice_ind:d}', 'data.mat')

                data = loadmat(data_path)
                im = data['im']
                TI = data['TI'] / time_scaling

                im = torch.tensor(im, dtype=torch.float32)
                im = torch.permute(im, [2, 0, 1])
                TI = torch.tensor(TI, dtype=torch.float32)
                TI = torch.squeeze(TI)

                im_record.append(im)
                TI_record.append(TI)

        self.im_record = im_record
        self.TI_record = TI_record
        self.ds_transforms = ds_transforms

    def __len__(self):
        return len(self.im_record)

    def __getitem__(self, ind):
        im = self.im_record[ind]
        TI = self.TI_record[ind]

        if self.ds_transforms:
            im = self.ds_transforms(im)

        return im, TI


class RegSegDataset(Dataset):
    # Dataset class for joint registration and segmentation
    def __init__(self, data_dir, sub_ind_record, slice_ind_record, im_shape, ds_transforms1, ds_transforms2):
        time_scaling = 1000

        im_record = []
        TI_record = []
        mask_record = []
        frame_order_record = []
        for sub_ind in sub_ind_record:

            data_dir2 = os.path.join(data_dir, f'subject{sub_ind:d}')
            for slice_ind in slice_ind_record:
                data_path = os.path.join(data_dir2, f'slice{slice_ind:d}', 'data.mat')
                frame_order_path = os.path.join(data_dir2, f'slice{slice_ind:d}', 'random_frame_order.mat')

                data = loadmat(data_path)
                im = data['im']
                TI = data['TI'] / time_scaling
                mask = data['mask']

                im = torch.tensor(im, dtype=torch.float32)
                im = torch.permute(im, [2, 0, 1])
                TI = torch.tensor(TI, dtype=torch.float32)
                TI = torch.squeeze(TI)
                mask = torch.tensor(mask, dtype=torch.float32)
                mask = torch.permute(mask, [2, 0, 1])

                frame_order = loadmat(frame_order_path)['frame_order'] - 1
                frame_order = torch.tensor(frame_order, dtype=torch.uint8)
                frame_order = torch.squeeze(frame_order)

                im_record.append(im)
                TI_record.append(TI)
                mask_record.append(mask)
                frame_order_record.append(frame_order)

        self.im_record = im_record
        self.TI_record = TI_record
        self.mask_record = mask_record
        self.frame_order_record = frame_order_record
        self.ds_transforms1 = ds_transforms1  # For all frames
        self.ds_transforms2 = ds_transforms2  # For every single frame
        self.center_crop = transforms.CenterCrop(size=im_shape)

    def __len__(self):
        return len(self.im_record)

    def __getitem__(self, ind):
        im = self.im_record[ind]
        TI = self.TI_record[ind]
        mask = self.mask_record[ind]
        frame_order = self.frame_order_record[ind]

        Nt = im.shape[0]

        # Augmentation for all frames
        if self.ds_transforms1:
            im_mask = self.ds_transforms1(torch.concatenate([im, mask], dim=0))
            im = im_mask[0:Nt, :, :]
            mask = im_mask[Nt:, :, :]

        # Augmentation for every single frame
        if self.ds_transforms2:
            for t in range(Nt):
                if torch.rand(1) < 0.3:
                    im_mask = self.ds_transforms2(
                        torch.concatenate([im[t:t + 1, :, :], mask[3 * t:3 * t + 3, :, :]], dim=0))
                    im[t:t + 1, :, :] = im_mask[0:1, :, :]
                    mask[3 * t:3 * t + 3, :, :] = im_mask[1:, :, :]

        # Center cropping
        im = self.center_crop(im)
        mask = self.center_crop(mask)

        # Binarization
        mask = torch.heaviside(mask - 0.5, values=torch.tensor(0.0))

        return im, TI, mask, frame_order
