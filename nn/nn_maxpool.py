# 最大池化

import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
# MaxPool2d要求输入像素点为浮点类型
input = torch.reshape(input, (-1, 1, 5, 5))

print(input.shape)


class MyMaxPool(nn.Module):
    def __init__(self):
        super(MyMaxPool, self).__init__()
        # ceil_mode决定kernel移动窗口部分移出图像时是否选择窗口部分进行最大池化
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output


dataset = torchvision.datasets.CIFAR10("../dataset",train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, num_workers=0)
writer = SummaryWriter("maxpool_logs")
my_max_pool = MyMaxPool()
step = 0

for data in dataloader:
    imgs, targets = data
    # print(type(imgs))
    output = my_max_pool(imgs)
    print(imgs.shape)
    # print(output.shape)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    # step = step + 1

writer.close()