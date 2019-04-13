# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by SHEN HUIXIANG  (shhuixi@qq.com)
# Created On: 2018-12-01
# --------------------------------------------------------
from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
import random
warnings.filterwarnings("ignore")
plt.ion()


class CloudDataset(Dataset):
    """Cloud Segmentation dataset"""

    def __init__(self, img_dir, labels_dir,val=False, transform=None):
        """
        Args:
            img_dir:Directory with all images.
            labels_dir:Directory with all Gt_maps.
            transform:Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.val=val

        img_names = os.listdir(self.img_dir)
        img_names.sort()

        lab_names = os.listdir(self.labels_dir)
        lab_names.sort()

        random.seed(32)
        random.shuffle(lab_names)
        random.seed(32)
        random.shuffle(img_names)
        self.train_img_names=img_names[:19000]
        self.train_lab_names=lab_names[:19000]
        self.val_img_names=img_names[19000:]
        self.val_lab_names=lab_names[19000:]

    def __len__(self):
        if self.val:
            return len(self.val_lab_names)
        else:
            return len(self.train_lab_names)

    def __getitem__(self, idx):
        if self.val:
            img_names=self.val_img_names
            lab_names=self.val_lab_names
        else:
            img_names=self.train_img_names
            lab_names=self.train_lab_names
        img_path = os.path.join(self.img_dir, img_names[idx])
        lab_path = os.path.join(self.labels_dir, lab_names[idx])

        image = io.imread(img_path)
        gt_map = io.imread(lab_path)
       
        sample = {'image': image, 'gt_map': gt_map / 255.0}
        if self.transform:
            sample = self.transform(sample)
        return sample


def show_image_GtMap(image, gt_map):
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(gt_map)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top:top + new_h, left:left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays is sample['image'],sample['gt_maps']"""

    def __call__(self, sample):
        image, gt_map = sample['image'], sample['gt_map']

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        gt_map = torch.from_numpy(gt_map)
        gt_map = torch.unsqueeze(gt_map,0)

        return {'image':image.type(torch.FloatTensor),
                'gt_map':gt_map.type(torch.float32)     
        }


if __name__ == '__main__':
    """测试本模块函数是否正常工作"""

    cloud_data = CloudDataset(
        img_dir='data/images224', labels_dir='data/masks224/')

    for i in range(len(cloud_data)):
        sample = cloud_data[i]
        print(i, sample['image'].shape, sample['gt_map'].shape)
        if i == 3:
            break
    tarnsformed_dataset = CloudDataset(
        img_dir='data/images224',
        labels_dir='data/masks224/',
        transform=transforms.Compose([ToTensor()]))
        
    for i in range(len(tarnsformed_dataset)):
        sample = tarnsformed_dataset[i]

        print(i, sample['image'].size(), sample['gt_map'].size())

        if i == 3:
            break

    dataloader = DataLoader(
        tarnsformed_dataset, batch_size=4, shuffle=True, num_workers=4)


    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['gt_map'].size())
        if i_batch == 3:
            break
