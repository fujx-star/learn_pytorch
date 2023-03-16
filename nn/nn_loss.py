# 损失函数
import PIL.Image
import torchvision
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor())
writer = SummaryWriter("loss_logs")
step = 0
# 差平均
loss1 = L1Loss(reduction='mean')
# 平方差
loss2 = MSELoss()
loss3 = CrossEntropyLoss()
img_start, target_start = dataset[1]
for i in range(10):
    img, target = dataset[i]
    print(loss1(img_start, img), end=' ')
    print(loss2(img_start, img), end=' ')
    print(loss3(img_start, img))
    writer.add_image("imgs", img, step)
    step = step + 1

