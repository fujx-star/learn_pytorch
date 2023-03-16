# 优化器（反向传播）
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, Flatten
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor

dataset = torchvision.datasets.CIFAR10("../dataset",train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
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
        result = self.model(x)
        return result


loss = nn.CrossEntropyLoss()
model = MyModel()
# lr为训练速度
learning_rate = 1e-2
optimizer = SGD(model.parameters(), lr=learning_rate)
test_image = torch.tensor(1)
test_target = torch.tensor(1)

for epoch in range(30):
    cur_loss = 0
    i = 0
    for data in dataloader:
        img, target = data
        if i > 100:
            test_image = img
            test_target = target
            break
        i = i+1
        output = model(img)
        loss_value = loss(output, target)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss_value.backward()
        # 更新卷积核
        optimizer.step()
        cur_loss = cur_loss + loss_value
    print(cur_loss)

transform = ToPILImage()
normal = torch.reshape(test_image, (3, 32, 32))
image = transform(normal)
output = model(test_image)
print(test_target)
print(output)
image.show()
print(dataset.class_to_idx)
# torch.save(model.state_dict(), "my_model_2.pth")




