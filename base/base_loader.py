# DataLoader将数据打包分块

import torchvision

# 准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

writer = SummaryWriter("base_loader")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    step = step + 1

writer.close()
