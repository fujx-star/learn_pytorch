# tensorboard基本操作

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("base_tensorboard")

for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)