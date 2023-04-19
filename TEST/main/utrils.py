import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from PIL import Image
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据预处理
data_transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

loss_list = []
val_loss_list = []
test_loss_list = []
val_acc_list = []
test_acc_list = []


def predict(image_path, model, topk=5):
    img = Image.open(image_path)
    img = img.unsqueeze(0)
    result = model(img.cuda()).topk(topk)
    probs = []
    classes = []
    a = result[0]
    b = result[1].tolist()

    for i in a[0]:
        probs.append(torch.exp(i).tolist())
    for n in b[0]:
        classes.append(str(n + 1))

    return (probs, classes)


# 绘制loss和acc图
def plot_loss_accuracy(loss_list, val_loss_list, test_loss_list, acc_list, test_acc_list, epoch):
    plt.figure()
    plt.plot(range(len(loss_list)), loss_list, label='running_loss')
    plt.plot(range(len(val_loss_list)), val_loss_list, label='val_loss')
    plt.plot(range(len(test_loss_list)), test_loss_list, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('mobilevit_20_5_fig_{}_loss'.format(epoch))

    plt.figure()
    plt.plot(range(len(acc_list)), acc_list, label='val_acc')
    plt.plot(range(len(test_acc_list)), test_acc_list, label='test_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('mobilevit_20_5_flg_{}_acc'.format(epoch))
    plt.show()


# 保存loss和acc为txt文件
def save_metrics_to_txt(loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list, file_path):
    with open(file_path, 'w') as f:
        f.write("Epoch, Loss, Val_Loss, Test_Loss, Val_Acc, Test_Acc\n")
        for i in range(len(loss_list)):
            f.write(
                f"{i + 1}, {loss_list[i]}, {val_loss_list[i]}, {test_loss_list[i]}, {val_acc_list[i]}, {test_acc_list[i]}\n")


# 训练模型
def train_model(model, criterion, optimizer, dataload_train, dataload_test, test_loader, num_epochs, batch_size, fold,
                weights_path, file_path):
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    for epoch in range(num_epochs):

        dataset_size = len(dataload_train.dataset)
        epoch_loss = 0
        step = 0
        with tqdm(total=dataset_size) as pbar:
            for imgs, labels in dataload_train:
                optimizer.zero_grad()
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                running_loss = criterion(outputs, labels)
                running_loss.backward()
                optimizer.step()
                # scheduler.step()
                epoch_loss += running_loss.item()

                pbar.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
                pbar.set_postfix(mini_loss=epoch_loss / (step + 1), epoch_loss=float(epoch_loss),
                                 running_loss=float(running_loss))
                pbar.update(batch_size)
                step += 1
            loss_list.append(epoch_loss / len(dataload_train))
            torch.save(model.state_dict(), weights_path)

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
        print('Accuracy of the network on the', len(dataload_test.dataset), 'val images: %d %%' % (
                100 * correct / total))

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
                total += labels.size(0)
                test_acc += torch.sum(preds == labels.data)

            test_loss /= len(test_loader)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc.item() / len(test_loader) * 100)
            print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_acc.item()/len(test_loader) ))
            print('the acc of the network on the ',len(test_loader.dataset),'test images: %d %%' % (
                100 * correct / total))

        plot_loss_accuracy(loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list, fold)
        save_metrics_to_txt(loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list, file_path=file_path)
    return model,loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list


# 训练
def train(criterion, optimizer, model, train_data_path, test_data_path, weights_path, batch_size, num_epochs,
          file_path):
    dataset = ImageFolder(train_data_path, transform=data_transform)
    dataset_test = ImageFolder(test_data_path, transform=data_transform)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    loss_list = []
    val_loss_list = []
    test_loss_list = []
    val_acc_list = []
    test_acc_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, dataset.targets)):
        print(f"Fold {fold + 1}:")

        train_sampler = Subset(dataset, train_idx)
        val_sampler = Subset(dataset, val_idx)

        dataload_train = DataLoader(train_sampler, batch_size=batch_size, shuffle=True, num_workers=4)
        dataload_test = DataLoader(val_sampler, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

        model, fold_loss_list, fold_val_loss_list, fold_test_loss_list, fold_val_acc_list, fold_test_acc_list = \
            train_model(model, criterion, optimizer, dataload_train, dataload_test, test_loader, num_epochs, batch_size,
                        fold, weights_path, file_path)

        loss_list.append(fold_loss_list)
        val_loss_list.append(fold_val_loss_list)
        test_loss_list.append(fold_test_loss_list)
        val_acc_list.append(fold_val_acc_list)
        test_acc_list.append(fold_test_acc_list)

        print(f"Finished training fold {fold + 1}")

    return model, loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list
