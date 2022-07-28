import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor()
])


def trans_image(root):
    img = Image.open(root).convert('L')  # model 'L'>>Grayscale image
    return img


class MyDataset(Dataset):
    def __init__(self, path):  # Initialization parameters
        self.path = path
        self.name = os.listdir(os.path.join(path, 'data'))  # xxx.png(label)--return a list

    def __len__(self):  # return len(img)
        return len(self.name)

    def __getitem__(self, index):  # Return the  data and labels according to the index
        segment_name = self.name[index]  # xx.png[index]
        segment_path = os.path.join(self.path, 'label', segment_name)  # Splicing path
        image_path = os.path.join(self.path, 'data', segment_name)
        image = np.array(Image.open(image_path).convert("L"))
        label = np.array(Image.open(segment_path).convert('L'))  # get transformed-files

        return transform(image), transform(label)
