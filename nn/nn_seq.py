# Sequential简化神经网络代码操作
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage, ToTensor
import PIL.Image


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 3*32*32
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        # 32*32*32
        self.max_pool1 = MaxPool2d(2)
        # 32*16*16
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        # 32*16*16
        self.max_pool2 = MaxPool2d(2)
        # 32*8*8
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        # 64*8*8
        self.max_pool3 = MaxPool2d(2)
        # 64*4*4
        self.flatten = Flatten()
        # 1024
        self.linear1 = Linear(1024, 64)
        # 64
        self.linear2 = Linear(64, 10)
        # 10

        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        output = self.model(x)
        return output


model = MyModel()
dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor())

# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# writer = SummaryWriter("seq_logs")
# step = 0
# for data in dataloader:
#     img, target = data
#     print(img.shape)
#     print(target)
#     output = model(img)
#     torch.reshape()
#     print(output)
#     step = step + 1
# img, target = dataset[0]
# transform = torchvision.transforms.ToTensor()
# tensor = transform(img)
# img.show()
# print(type(img))
# print(type(tensor))
# transform1 = ToPILImage()
# data = transform1(tensor)
# data.show()


data, target = dataset[0]
input = torch.reshape(data, (-1, 3, 32, 32))
output = model(input)

writer = SummaryWriter("seq_logs")
writer.add_image("input", data)
writer.add_graph(model, input)
writer.close()

