import torch
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import data
import resnet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    plt.savefig('res_test_fig_test_{}_loss_1'.format(epoch))

    # 绘制 accuracy 图表
    plt.figure()
    plt.plot(range(len(acc_list)), acc_list, label='val_acc')
    plt.plot(range(len(test_acc_list)), test_acc_list, label='test_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('res_test_flg_test_{}_acc_1'.format(epoch))
    plt.show()


def evaluate(model, criterion, dataload_test):
    model.eval()  # 设置模型为评估模式

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    with torch.no_grad():  # 关闭梯度计算
        for imgs, labels in dataload_test:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)  # 计算loss总和
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()  # 计算正确预测的数量
            total_num += imgs.size(0)  # 计算总的样本数量

    avg_loss = total_loss / total_num  # 计算平均loss
    accuracy = total_correct / total_num  # 计算准确率

    return avg_loss, accuracy


def train_model(model, criterion, optimizer, dataload_train, dataload_test, test_loader, num_epochs, batch_size, flod):
    # model:模型, criterion:损失函数, optimizer:优化器

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')
    loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list = [], [], [], [], []

    for epoch in range(num_epochs):

        dataset_size = len(dataload_train.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        model.train()
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

            # 计算验证集准确率
            val_loss, val_accuracy = evaluate(model, criterion, dataload_test)
            print('val loss: {:.2f}'.format(val_loss))
            print('Validation accuracy: {:.2f}%'.format(val_accuracy * 100))

            # 计算测试集准确率
            test_loss, test_accuracy = evaluate(model, criterion, test_loader)
            print('test loss: {:.2f}'.format(test_loss))
            print('Test accuracy: {:.2f}'.format(test_accuracy * 100))

        plot_loss_accuracy(loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list, flod)
        # 为每个列表创建一个标签
        labels = ['train_loss:', 'val_loss:', 'test_loss:', 'val_acc:', 'test_acc:']

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
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 梯度下降
    optimizer = optim.SGD(model.parameters(),
                          lr=1e-4)  # model.parameters():Returns an iterator over module parameters,The default is 1e-3
    # 加载数据集
    train_loader, val_loader, test_loader = data.load_data(batch_size, num_workers,train_data_path=train_data_pth,test_data_path=test_data_pth)
    fold = 1
    train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, num_epochs, batch_size, fold)
    # model :训练模型
    # criterion : 损失函数
    # optimizer : 优化器
    # train_loader : 训练集
    # val_loader : 验证集
    # test_loader: 测试集


if __name__ == '__main__':
    model = resnet.resnet50(2)
    model = model.cuda()
    weight_path = r'D:\desktop\sick_code\ZZ\params\resnet_2_100_1_test.pth'
    train_data_pth = r'E:\sick_datasets\z\train'
    test_data_pth = r'E:\sick_datasets\z\test'
    num_epochs = 100
    batch_size = 6
    num_workers = 2
    train(model)
