# 非线性激活函数
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, num_workers=0)
writer = SummaryWriter("nonlinear_logs")


# 线性变换
class NonLinear(nn.Module):
    def __init__(self):
        super(NonLinear, self).__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        relu_output = self.relu(input)
        sigmoid_output = self.sigmoid(input)
        return relu_output, sigmoid_output


step = 0
linear = NonLinear()

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    relu_output, sigmoid_output = linear(imgs)
    writer.add_images("relu_output", relu_output, global_step=step)
    writer.add_images("sigmoid_output", sigmoid_output, global_step=step)
    step = step + 1
writer.close()
