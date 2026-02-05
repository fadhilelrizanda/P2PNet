import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io

class SHHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False, patch_size=128, num_patch=4):
        self.root_path = data_root
        self.train_lists = "train.list"
        self.eval_list = "test.list"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}
        self.img_list = []
        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            list_path = os.path.join(self.root_path, train_list)
            if not os.path.exists(list_path):
                raise FileNotFoundError(f"List file not found: {list_path}")
            with open(list_path) as fin:
                for line in fin:
                    if len(line.strip()) == 0:
                        continue
                    parts = line.strip().split()  # robust whitespace split
                    if len(parts) < 2:
                        continue
                    img_rel, gt_rel = parts[0].strip(), parts[1].strip()
                    img_abs = os.path.join(self.root_path, img_rel)
                    gt_abs = os.path.join(self.root_path, gt_rel)
                    self.img_map[img_abs] = gt_abs
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip
        self.patch_size = patch_size
        self.num_patch = num_patch

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # load image and ground truth
        img, points = load_data((img_path, gt_path), self.train)
        # img is a PIL.Image, points is numpy (N,2)

        # apply transform if provided (should produce a torch.Tensor C,H,W)
        if self.transform is not None:
            img = self.transform(img)

        # ensure img is a torch.Tensor
        if not isinstance(img, torch.Tensor):
            img = torch.Tensor(np.array(img)).permute(2, 0, 1).float() / 255.0

        # augmentation: random scale
        if self.train:
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])  # H, W
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > self.patch_size:
                # upsample/downsample image tensor (use interpolate)
                img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
                points = points * scale  # numpy array multiplied by scalar

        # random crop augmentation (produce patches)
        if self.train and self.patch:
            img_patches, patch_points = random_crop(img, points, patch_h=self.patch_size, patch_w=self.patch_size, num_patch=self.num_patch)
            # img_patches: torch.Tensor (num_patch, C, H, W)
            # patch_points: list of numpy arrays per patch
            # convert points to list of tensors
            points_list = [torch.Tensor(p) if p.size else torch.zeros((0,2)) for p in patch_points]
            img = img_patches  # shape (num_patch, C, H, W)
        else:
            # No patching: keep single image and wrap points in list for consistent target format
            points_list = [torch.Tensor(points) if points.size else torch.zeros((0,2))]

        # random flipping (horizontal)
        if self.train and self.flip:
            if self.patch:
                # flip each patch and update coordinates using torch.flip
                img = torch.flip(img, dims=[-1]).contiguous()  # (num_patch, C, H, W) flip last dim
                pw = img.shape[-1]
                for i, p in enumerate(points_list):
                    if p.numel() == 0:
                        continue
                    # p is tensor shape (M,2) with x in column 0
                    p[:, 0] = (pw - 1) - p[:, 0]
                    points_list[i] = p
            else:
                # single image flip
                img = torch.flip(img, dims=[-1]).contiguous()  # (C, H, W)
                w = img.shape[-1]
                p = points_list[0]
                if p.numel() != 0:
                    p[:, 0] = (w - 1) - p[:, 0]
                    points_list[0] = p

        # prepare target list (one dict per patch or per image if no patch)
        target = []
        for i, pts in enumerate(points_list):
            t = {}
            # ensure pts is a torch.Tensor of shape (N,2)
            if not isinstance(pts, torch.Tensor):
                pts = torch.Tensor(pts)
            t['point'] = pts
            # image id: try robust extraction, fallback to index
            try:
                image_id = int(os.path.basename(img_path).split('.')[0].split('_')[-1])
            except Exception:
                image_id = index
            t['image_id'] = torch.Tensor([image_id]).long()
            t['labels'] = torch.ones([pts.shape[0]]).long() if pts.shape[0] > 0 else torch.zeros([0]).long()
            target.append(t)

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images using cv2 then convert to PIL (to be compatible with transforms)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            line = line.strip()
            if not line:
                continue
            parts = line.split()  # robust split
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                points.append([x, y])
            except:
                continue

    return img, np.array(points)


# random crop augmentation
def random_crop(img, den, patch_h=128, patch_w=128, num_patch=4):
    """
    img: torch.Tensor (C, H, W)
    den: numpy array (N, 2) (x, y)
    returns:
      img_patches: torch.Tensor (num_patch, C, patch_h, patch_w)
      patch_points: list of numpy arrays (per patch)
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("random_crop expects img to be a torch.Tensor")

    C, H, W = img.shape
    if H < patch_h or W < patch_w:
        # If the image is smaller than patch, we can pad it
        pad_h = max(0, patch_h - H)
        pad_w = max(0, patch_w - W)
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        C, H, W = img.shape

    patches = []
    patch_points = []

    for i in range(num_patch):
        start_h = random.randint(0, H - patch_h)
        start_w = random.randint(0, W - patch_w)
        end_h = start_h + patch_h
        end_w = start_w + patch_w
        patch = img[:, start_h:end_h, start_w:end_w].clone()
        patches.append(patch)

        if den is None or den.size == 0:
            patch_points.append(np.zeros((0,2)))
            continue

        # Select points inside the crop (inclusive left/top, exclusive right/bottom)
        mask = (den[:, 0] >= start_w) & (den[:, 0] < end_w) & (den[:, 1] >= start_h) & (den[:, 1] < end_h)
        selected = den[mask].copy()
        if selected.size:
            selected[:, 0] -= start_w
            selected[:, 1] -= start_h
        patch_points.append(selected)

    img_patches = torch.stack(patches, dim=0)  # (num_patch, C, H, W)
    return img_patches, patch_points