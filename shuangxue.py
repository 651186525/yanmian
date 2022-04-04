import os
import glob
import random
import numpy as np
import cv2
from pandas import lreshape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.unet import UNet
from torch.utils.tensorboard import SummaryWriter
import time


def get_max_indices(heatmap):
    assert heatmap.dim() in [2, 3, 4]
    if heatmap.dim() == 2:
        m = torch.argmax(heatmap).item() + 1
        width, height = heatmap.size()
        import math
        x = math.ceil(m / height) - 1
        y = m - x * height - 1
        return np.asarray([x, y])
    else:
        indices_list = []
        for i in range(heatmap.size(0)):
            indices_list.append(get_max_indices(heatmap[i]).tolist())
        return np.asarray(indices_list)


class LandmarksDataset(Dataset):
    def __init__(self, img_root, annotation_root, file_names, out_size, aug=False):
        self.img_root = img_root
        self.annotation_root = annotation_root
        self.file_names = file_names
        self.out_size = out_size
        self.aug = aug

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # load data
        img_path = glob.glob(os.path.join(self.img_root, f"{self.file_names[idx]}.*"))[0]
        hm_gt_path = os.path.join(self.annotation_root, f"{self.file_names[idx]}.npy")
        img = cv2.imread(img_path)
        hm_gt = np.load(hm_gt_path)
        img = torch.as_tensor(img, dtype=torch.float32)
        hm_gt = torch.as_tensor(hm_gt, dtype=torch.float32)
        # hwc2chw
        img = img.permute(2, 0, 1)
        hm_gt = hm_gt.permute(2, 0, 1)
        # resize image

        if self.aug:
            # random flip
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])
                hm_gt = torch.flip(hm_gt, dims=[2])

        # resize and pad : resize
        new_size = self.out_size
        nh, nw = new_size
        ih, iw = img.size()[-2:]
        resize_ratio = min(nh / ih, nw / iw)
        interpolate_size = (
            round(resize_ratio * ih),
            round(resize_ratio * iw),
        )
        img = img.unsqueeze(0)
        img = F.interpolate(img, size=interpolate_size, mode='bilinear', align_corners=True)
        img = img.squeeze(0)
        hm_gt = F.interpolate(hm_gt.unsqueeze(0), size=interpolate_size, mode='bilinear', align_corners=True).squeeze(0)
        # resize and pad : pad
        ch, cw = new_size
        ih, iw = interpolate_size
        oh, ow = min(ih, ch), min(iw, cw)
        img = img[:, :oh, :ow]
        img = F.pad(img, pad=(0, cw - ow, 0, ch - oh))
        hm_gt = hm_gt[:, :oh, :ow]
        hm_gt = F.pad(hm_gt, pad=(0, cw - ow, 0, ch - oh))

        # return {'image':img,
        #         'mask' :hm_gt,
        #         'resize_ratio':resize_ratio
        # }
        return img, hm_gt, resize_ratio, self.file_names[idx]


if __name__ == '__main__':
    device = torch.device('cuda:2')
    batch_size = 32
    epochs = 100
    var = 15
    data_name = 'face_lk'
    out_size = [256, 384]
    img_root_train = '/data/sx/lm/code_nt/data_lk/face_469/train/img'
    img_root_test = '/data/sx/lm/code_nt/data_lk/face_469/test/img'
    annotation_root_train = img_root_train.replace('img', f'np_var{var}')
    annotation_root_test = img_root_test.replace('img', f'np_var{var}')
    file_names_train = [file.split(".")[0] for file in os.listdir(img_root_train)]
    file_names_test = [file.split(".")[0] for file in os.listdir(img_root_test)]
    dataset_train = LandmarksDataset(img_root_train, annotation_root_train, file_names_train, out_size,
                                     aug=True)  # out_size=(h, w)
    dataset_test = LandmarksDataset(img_root_test, annotation_root_test, file_names_test, out_size)
    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=16)
    dl_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=16)

    lr = 2e-4
    writer = SummaryWriter(
        comment=f'LR_{lr}_BS_{batch_size}_VAR_{var}_DATANUM_{str(len(file_names_train) + len(file_names_test))}_DATANAME_{data_name}_{out_size}')
    global_step = 0

    model = UNet(in_channels=3, num_classes=6, bilinear=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(size_average=True).to(device)
    trainning_metrics = []
    for eph in range(epochs):
        model.train()
        losses = []
        errs_all = []
        for img, hm_gt, resize_ratio, img_name in dl_train:
            img = img.to(device)
            hm_gt = hm_gt.to(device)
            output = model(img)
            loss = criterion(output, hm_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # global_step += 1
            with torch.no_grad():
                # land_gt = get_max_indices(hm_gt) / resize_ratio.numpy().reshape(-1)
                # land_pred = get_max_indices(output) / resize_ratio.numpy().reshape(-1)
                land_gt = get_max_indices(hm_gt) / resize_ratio.unsqueeze(1).unsqueeze(1).numpy()
                land_pred = get_max_indices(output) / resize_ratio.unsqueeze(1).unsqueeze(1).numpy()
                errs = np.sqrt(np.sum(np.square(land_pred - land_gt), axis=2)).tolist()
            losses.append(loss.item())
            errs_all += errs
        losses_mean_train = sum(losses) / len(losses)
        errs_mean_train = [sum(m) / len(m) for m in zip(*errs_all)]
        #
        writer.add_scalar('Loss/train', losses_mean_train, eph)
        writer.add_scalar('Errs1/train', errs_mean_train[0], eph)
        writer.add_scalar('Errs2/train', errs_mean_train[1], eph)
        writer.add_scalar('Errs3/train', errs_mean_train[1], eph)
        writer.add_scalar('Errs4/train', errs_mean_train[1], eph)
        writer.add_scalar('Errs5/train', errs_mean_train[1], eph)
        writer.add_scalar('Errs6/train', errs_mean_train[1], eph)
        writer.add_scalar('Errs1 + Errs2 + Errs3 + Errs4 + Errs5 + Errs6/train', sum(errs_mean_train), eph)

        model.eval()
        errs_all = []
        for img, hm_gt, resize_ratio, img_name in dl_test:
            img = img.to(device)
            hm_gt = hm_gt.to(device)
            with torch.no_grad():
                output = model(img)
                # land_gt = get_max_indices(hm_gt) / resize_ratio.numpy().reshape(-1)
                # land_pred = get_max_indices(output) / resize_ratio.numpy().reshape(-1)
                land_gt = get_max_indices(hm_gt) / resize_ratio.unsqueeze(1).unsqueeze(1).numpy()
                land_pred = get_max_indices(output) / resize_ratio.unsqueeze(1).unsqueeze(1).numpy()
                errs = np.sqrt(np.sum(np.square(land_pred - land_gt), axis=2)).tolist()
            errs_all += errs
        errs_mean_test = [sum(m) / len(m) for m in zip(*errs_all)]
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        pt_save_dir = f"ckpts_aug/LR_{lr}_BS_{batch_size}_VAR_{var}_DATANUM_{str(len(file_names_train) + len(file_names_test))}_DATANAME_{data_name}_{out_size}"
        if os.path.exists(pt_save_dir):
            pass
        else:
            os.mkdir(pt_save_dir)
        torch.save(model.state_dict(), f"{pt_save_dir}/epoch{eph}_date{date}.pt")
        writer.add_scalar(
            'learning_rate', optimizer.param_groups[0]['lr'], eph)
        writer.add_scalar('Errs1/val', errs_mean_test[0], eph)
        writer.add_scalar('Errs2/val', errs_mean_test[1], eph)
        writer.add_scalar('Errs3/val', errs_mean_test[1], eph)
        writer.add_scalar('Errs4/val', errs_mean_test[1], eph)
        writer.add_scalar('Errs5/val', errs_mean_test[1], eph)
        writer.add_scalar('Errs6/val', errs_mean_test[1], eph)
        writer.add_scalar('Errs1 + Errs2 + Errs3 + Errs4 + Errs5 + Errs6/val', sum(errs_mean_test), eph)
        print(eph, losses_mean_train, errs_mean_train, errs_mean_test)
    writer.close()
