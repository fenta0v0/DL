import argparse
import os
import time

import torch
import Unet
from torch import optim
from PIL import Image
from torchvision.transforms import transforms, Compose
from torchvision.utils import save_image
from Data import MyDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import res_unet
import datetime

# 设置参数

"""
parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象
parser.add_argument('action', type=str, help='train or test')  # 添加参数
parser.add_argument('--batch_size', type=int, default=4, help='Number of epochs to train.')
parser.add_argument('--weight', type=str, help='the path of the mode weight file')
parser.add_argument('--weight_path', type=str, help='the path of weight')
parser.add_argument('')
args = parser.parse_args()
"""
"""trans_ops = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])"""
# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_transform: Compose = transforms.Compose([
    transforms.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # torchvision.transforms.Normalize(mean, std, inplace=False)
])
# label只需要转换为tensor
y_transform = transforms.ToTensor()
pth = r'D:\python project\DL\NET\params\unet.pth'
weight_path = pth  # 权重地址

writer = SummaryWriter(log_dir="logs", flush_secs=60)  # 60s写入一次


def train_model(model, criterion, optimizer, dataload, num_epochs):
    # model:模型, criterion:损失函数, optimizer:优化器

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        for img, label in dataload:  #
            optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
            imgs = img.to(device)
            labels = label.to(device)
            outputs = model(imgs)  # 前向传播
            running_loss = criterion(outputs, labels)
            running_loss.backward()  # 梯度下降,计算出梯度
            optimizer.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += running_loss.item()
            step += 1
            if step % 20 == 19:  # every 1000 mini-batches...

                # ...log the running loss
                writer.add_scalar('training loss',
                                  running_loss,
                                  epoch * len(dataload) + step)

            print("%d/%d,train_loss:%0.4f" % (step, dataset_size // dataload.batch_size, running_loss.item()))
            running_loss = 0
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        writer.add_scalar('epoch loss', epoch_loss, epoch)
        torch.save(model.state_dict(), weight_path)  # 返回模型的所有内容
    return model


# 训练模型
def train(model):
    model = model
    # model = res_unet.NET(1, 1).to(device)  # 加载模型
    num_epochs = 30
    batch_size = 3
    # 损失函数
    criterion = torch.nn.BCELoss()
    # 梯度下降
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-4)  # model.parameters():Returns an iterator over module parameters,The default is 1e-3
    # 加载数据集
    liver_dataset = MyDataset('D:/image/train')
    dataloader = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # batch_size：how many samples per minibatch to load，
    # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度
    train_model(model, criterion, optimizer, dataloader, num_epochs)


# 测试


def test(model):
    var_path = r'D:\image\val\data'
    model = model
    # model = res_unet.NET(1, 1).cuda()  # 加载模型
    model.load_state_dict(torch.load(weight_path))  # 加载权重
    var_name = os.listdir(os.path.join(var_path))
    for i in range(len(var_name)):
        image_path = os.path.join(var_path, "%d.png" % (i + 1))
        img = Image.open(image_path)
        if img.mode != 'L':
            img = img.convert('L')
        x = y_transform(img).cuda()
        mask = torch.unsqueeze(x, dim=0)
        out = model(mask)
        save_image(out, r'D:\image\val\RS_label\res_unet\% d.png' % (i + 1))
        print(i)


"""def test():
    model = Unet.UNet(1, 1)  # 加载模型
    model.load_state_dict(torch.load(weight_path))  # 加载权重
    liver_dataset = MyDataset("E:/Project image/val")
    dataloaders = DataLoader(liver_dataset)  # batch_size默认为1
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        i = 0
        for x, _ in dataloaders:   # 获取data>>>label
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            plt.imshow(img_y)
            plt.savefig('E:/Project image/val/test_image/% d.png' % (i + 1))
            i += 1
"""
if __name__ == '__main__':
    stat_time = time.time()
    model = Unet.UNet(1, 1).cuda()
    """print('please input train or test')
    x = input()
    if x == str('train'):
        train(model)
    else:
        test(model)"""
    train(model)
    total_time = time.time() - stat_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

"""
if args.action == 'train':
    train()
elif args.action == 'test':
    test()"""
