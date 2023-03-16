# 用tranforms函数改变image的格式

from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("base_transform")
img = Image.open("../dataset/train/ants/6743948_2b8c096dda.jpg")
print(img)

trans_to_tensor = transforms.ToTensor()
img_tensor = trans_to_tensor(img)
writer.add_image("ToTensor", img_tensor)

print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6, 3, 2], [9, 3, 5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 5)
