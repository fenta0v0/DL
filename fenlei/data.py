import os
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

species = {'0': 0, '1': 1}
transform = transforms.Compose([transforms.ToTensor()])

class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        # root :'mnist\\test' or 'mnist\\train'
        self.root = root
        self.transform = transform
        self.data = []
        # 获取子目录 '0','1','2','3',...
        sub_root_test = os.listdir(self.root)
        for sub_root in sub_root_test:
            # 获取子目录下所有图片的名字
            sub_image_name_list = os.listdir(os.path.join(self.root, sub_root))
            for sub_image_name in sub_image_name_list:
                # 获取每张图片的完整路径
                image_path = os.path.join(self.root, os.path.join(sub_root, sub_image_name))
                # 获取标签，也就是子目录的文件名
                label = species[image_path.split('\\')[-2]]
                # 做成（图片路径，标签）的元组
                sample = (image_path, label)
                self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image_original = Image.open(image_path).convert('L')
        image_tensor = transform(image_original)
        return image_tensor, label


if __name__ == '__main__':
    data_dir = r'F:\dataset\train'


    dataset = MyDataset(data_dir)
    print(dataset[1000][0])
    print(dataset[1000][1])
