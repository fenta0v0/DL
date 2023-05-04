import torch
import torchvision.transforms as transforms
import torch.utils.data
from torchvision.datasets import ImageFolder


def load_data(batch_size, num_workers, train_data_path, test_data_path):
    # define the transformations for the images
    transform_train = transforms.Compose(
        [   transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose(
        [   transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load dataset
    trainset = ImageFolder(train_data_path, transform=transform_train)
    testset = ImageFolder(test_data_path, transform=transform_test)

    # split the trainset into train and validation sets
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(num_train * 0.8)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])

    # create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              sampler=train_sampler, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            sampler=val_sampler, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader


if __name__ == '__main__':
    trainloader, valloader, testloader = load_data(4, 4, r'E:\sick_datasets/z/train',
                                                           r'E:\sick_datasets/z/test')

    # 显示训练集和测试集的样本数量
    print('训练集样本数量：', len(trainloader.dataset))
    print('测试集样本数量：', len(testloader.dataset))
    print(len(valloader.dataset))

    # 获取一批数据
    data, label = next(iter(trainloader))
    # 显示一批数据的形状和标签
    print('数据形状：', data.shape)
    print('标签：', label)
