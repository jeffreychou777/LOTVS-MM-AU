import cv2
import numpy as np


def compute_mean_std_iter(img_list):
    means = np.zeros(3)
    std = np.zeros(3)
    n = 1
    for img_name in img_list:
        img = cv2.imread(img_name)
        for idx_channel in range(3):
            std[idx_channel] = (n-1)/n*std[idx_channel] + np.mean((img[:,:,idx_channel]-means[idx_channel])**2)/n - ((np.mean(img[:,:,idx_channel].ravel())-means[idx_channel])**2)/(n**2)
            means[idx_channel] = means[idx_channel] + (np.mean(img[:,:,idx_channel].ravel())-means[idx_channel])/n
        n = n + 1
    std = [np.sqrt(item) for item in std]
    return means,std


from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms


def get_mean_and_std(train_data):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    transform_train = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                                        saturation=0.2, hue=0.1)], p=0.8),
         transforms.ToTensor(),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
         transforms.GaussianBlur(kernel_size=3, sigma=(0.05, 1.0))
         ]
    )
    train_dataset = ImageFolder(root="/home/ubuntu/CAP-DATA/training", transform=transform_train)
    print(get_mean_and_std(train_dataset))

    # ([0.403932, 0.42002773, 0.42598072], [0.25544345, 0.2625126, 0.27170303])

