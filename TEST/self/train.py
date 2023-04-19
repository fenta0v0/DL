import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torchvision.transforms import transforms, Compose
from data1 import MyDataset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from PIL import Image

import resnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据预处理
data_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

x_transform: Compose = transforms.Compose([
    transforms.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # torchvision.transforms.Normalize(mean, std, inplace=False)
])
# label只需要转换为tensor
y_transform = transforms.ToTensor()
pth = r'D:\desktop\sick_code\params\resnet_2_1.pth'
weight_path = pth  # 权重地址


def predict(image_path, model, topk=5):
    """
    预测图片.
    """
    img = Image.open(image_path)
    img = img.unsqueeze(0)  # 将图片多增加一维
    result = model(img.cuda()).topk(topk)
    probs = []
    classes = []
    a = result[0]  # 返回TOPK函数截取的排名前列的结果列表a
    b = result[1].tolist()  # 返回TOPK函数截取的排名前列的概率索引列表b

    for i in a[0]:
        probs.append(torch.exp(i).tolist())  # 将结果转化为实际概率
    for n in b[0]:
        classes.append(str(n + 1))  # 将索引转化为实际编号

    return (probs, classes)


loss_list = []
val_loss_list = []
test_loss_list = []
val_acc_list = []
test_acc_list = []


def plot_loss_accuracy(loss_list, val_loss_list, test_loss_list, acc_list, test_acc_list, epoch):
    # 绘制 loss 图表
    plt.figure()
    plt.plot(range(len(loss_list)), loss_list, label='running_loss')
    plt.plot(range(len(val_loss_list)), val_loss_list, label='val_loss')
    plt.plot(range(len(test_loss_list)), test_loss_list, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('res_test_fig_{}_loss_1'.format(epoch))

    # 绘制 accuracy 图表
    plt.figure()
    plt.plot(range(len(acc_list)), acc_list, label='val_acc')
    plt.plot(range(len(test_acc_list)), test_acc_list, label='test_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('res_test_flg_{}_acc_1'.format(epoch))
    plt.show()


def train_model(model, criterion, optimizer, dataload_train, dataload_test, test_loader, num_epochs, batch_size, flod):
    # model:模型, criterion:损失函数, optimizer:优化器

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    for epoch in range(num_epochs):

        dataset_size = len(dataload_train.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        with tqdm(total=dataset_size) as pbar:
            # desc=f'Epoch{epoch + 1}/{num_epochs}', unit='it'

            for imgs, labels in dataload_train:  #

                optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)  # 前向传播
                running_loss = criterion(outputs, labels)
                running_loss.backward()  # 梯度下降,计算出梯度
                optimizer.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
                epoch_loss += running_loss.item()

                pbar.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
                pbar.set_postfix(mini_loss=epoch_loss / (step + 1), epoch_loss=float(epoch_loss),
                                 running_loss=float(running_loss))
                pbar.update(batch_size)
                step += 1
            loss_list.append(epoch_loss / len(dataload_train))
            torch.save(model.state_dict(), weight_path)  # 返回模型的所有内容

        # 准确率验证
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataload_test):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_loss += val_loss.item()


                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss_list.append(float(val_loss) / len(dataload_test))
        val_acc_list.append(100 * correct / total)
        print('Accuracy of the network on the', len(dataload_test.dataset), 'test images: %d %%' % (
                100 * correct / total))

        # 测试
        test_loss = 0.0
        test_acc = 0.0
        model.eval()

        with torch.no_grad():
            for i, (data, labels) in enumerate(test_loader):
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * data.size(0)
                _, preds = torch.max(outputs, 1)
                test_acc += torch.sum(preds == labels.data)

            test_loss /= len(test_loader)
            # test_acc /= len(test_loader)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc.item()/len(test_loader)*100)
            print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_acc.item()/len(test_loader)))

        plot_loss_accuracy(loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list, flod)
        # 为每个列表创建一个标签
        labels = ['train_loss:', 'val_loss:', 'test_loss:','val_acc:','test_acc:']

        # 将列表和标签放入另一个列表
        all_lists = list(zip(labels, [loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list]))

        # 遍历列表并将每个列表及其标签保存到单独的文本文件
        for idx, (label, current_list) in enumerate(all_lists):
            with open(f'D:/desktop/sick_code/txt/resnet/{flod}_list_test_{idx}.txt', 'w') as f:
                # 在列表之前写入标签
                f.write(f"{label}\n")

                # 写入列表内容
                for item in current_list:
                    f.write(f"{item}\n")
    return model


def train(model1):
    model = model1  # load model

    num_epochs = 20
    batch_size = 4
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 梯度下降
    optimizer = optim.SGD(model.parameters(),
                          lr=1e-4)  # model.parameters():Returns an iterator over module parameters,The default is 1e-3
    # 加载数据集
    # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # batch_size：how many samples per minibatch to load，
    # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度

    dataset = ImageFolder(r'E:\sick_datasets\z\train', transform=data_transform)
    dataset_test = ImageFolder(r'E:\sick_datasets\z\test', transform=data_transform)
    # 定义交叉验证折叠数
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, dataset.targets)):
        print(f"Fold {fold + 1}:")

        # 根据当前折叠划分训练集和验证集
        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, val_idx)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        test_loader = DataLoader(dataset_test)

        train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, num_epochs, batch_size, fold)
        # model :训练模型
        # criterion : 损失函数
        # optimizer : 优化器
        # train_loader : 训练集
        # val_loader : 验证集
        # test_loader: 测试集


if __name__ == '__main__':
    model = resnet.resnet50(5)
    model = model.cuda()
    train(model)
