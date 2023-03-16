# 二维卷积改变图像

import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True, shuffle=False)


class MyConv2d(nn.Module):
    def __init__(self):
        super(MyConv2d, self).__init__()
        self.conv = Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=2)

    def forward(self, x):
        return self.conv(x)


myconv_2d = MyConv2d()
writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    output = myconv_2d(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
