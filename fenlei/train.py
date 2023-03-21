import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms, Compose
from data import MyDataset
from net import net
from tqdm import tqdm
from PIL import Image
import net_33

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_transform: Compose = transforms.Compose([
    transforms.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # torchvision.transforms.Normalize(mean, std, inplace=False)
])
# label只需要转换为tensor
y_transform = transforms.ToTensor()
pth = r'E:\sick_code\params\test_1_net33.pth'
weight_path = pth  # 权重地址


# 验证准确率
def accuracy_test(model, dataloader):
    correct = 0
    total = 0
    model.cuda()  # 将模型放入GPU计算，能极大加快运算速度
    with torch.no_grad():  # 使用验证集时关闭梯度计算
        for data in dataloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            # 将预测及标签两相同大小张量逐一比较各相同元素的个数
    print('the accuracy is {:.4f}'.format(correct / total))

def predict(image_path, model, topk=5):
    ''' 预测图片.
    '''
    img = Image.open(image_path)
    img = img.unsqueeze(0)   # 将图片多增加一维
    result = model(img.cuda()).topk(topk)
    probs= []
    classes = []
    a = result[0]     # 返回TOPK函数截取的排名前列的结果列表a
    b = result[1].tolist() #返回TOPK函数截取的排名前列的概率索引列表b

    for i in a[0]:
        probs.append(torch.exp(i).tolist())  #将结果转化为实际概率
    for n in b[0]:
        classes.append(str(n+1))      # 将索引转化为实际编号

    return(probs,classes)

def train_model(model, criterion, optimizer, dataload_train,dataload_test, num_epochs, batch_size):
    # model:模型, criterion:损失函数, optimizer:优化器

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        dataset_size = len(dataload_train.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        with tqdm(total=dataset_size) as pbar:
            # desc=f'Epoch{epoch + 1}/{num_epochs}', unit='it'

            for imgs,labels in dataload_train:  #

                optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)  # 前向传播
                #labels = labels.unsqueeze(0)

                running_loss = criterion(outputs, labels)
                # running_loss = simm_loss(outputs,labels)
                running_loss.backward()  # 梯度下降,计算出梯度
                optimizer.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
                epoch_loss += running_loss.item()
                # acc = calculate_accuracy(model, dataload)
                pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
                pbar.set_postfix(mini_loss=epoch_loss / (step + 1), epoch_loss=float(epoch_loss),
                                 running_loss=float(running_loss))
                pbar.update(batch_size)
                step += 1
                """if step % 50 == 0:
                    # test the accuracy

                    print('EPOCHS : {}/{}'.format(epoch+1, num_epochs),
                          'Loss : {:.4f}'.format(epoch_loss / 50))"""
                    #accuracy_test(model, dataload)

            torch.save(model.state_dict(), weight_path)  # 返回模型的所有内容




        # 准确率验证
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataload_test):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the', len(dataload_test.dataset), 'test images: %d %%' % (
                100 * correct / total))

    return model

def train(model1):
    model = model1
    # model = res_unet.NET(1, 1).to(device)  # 加载模型
    num_epochs = 20
    batch_size = 4
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 梯度下降
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-5)  # model.parameters():Returns an iterator over module parameters,The default is 1e-3
    # 加载数据集
    train_dataset = MyDataset(r'D:\desktop\train\train')
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset = MyDataset(r'D:\desktop\train\test')
    dataloader_test = DataLoader(test_dataset,2)
    # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # batch_size：how many samples per minibatch to load，
    # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度]

    train_model(model, criterion, optimizer, dataloader_train,dataloader_test, num_epochs, batch_size)

def test(model):
    model = model
    model = model.cuda()
    input_path =r"D:\desktop\datasets\sick\data\10.png"
    img = Image.open(input_path)
    if img.mode != 'L':
        img = img.convert('L')
    x = y_transform(img).cuda()
    mask = torch.unsqueeze(x, dim=0)
    out = model(mask)
    print(out)

if __name__ == '__main__':
    model = net_33.net().cuda()
    train(model)
