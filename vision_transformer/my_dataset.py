from PIL import Image
import torch
from torch.utils.data import Dataset


from torchvision import transforms


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        # 如果传入的transform为None，则使用默认的预处理方式
        if transform is None:
            self.transform = transforms.Compose([
                # 调整图像大小，例如将图像统一调整为224x224大小，这是很多预训练模型常用的输入尺寸
                transforms.Resize((224, 224)),
                # 从图像中心裁剪出指定大小，此处也是裁剪为224x224，可保证图像比例合适
                transforms.CenterCrop(224),
                # 随机水平翻转，以进行数据增强（仅训练时常用）
                transforms.RandomHorizontalFlip(),
                # 将图像转换为张量，这是PyTorch模型输入所要求的格式
                transforms.ToTensor(),
                # 对图像进行归一化，这里使用的均值和标准差是常见的在ImageNet数据集上预训练时采用的值
                # 你可以根据自己训练数据的实际情况进行调整
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
