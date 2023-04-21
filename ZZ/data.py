import torch
import torchvision.transforms as transforms
import torch.utils.data
from torchvision.datasets import ImageFolder


def load_data(batch_size, num_workers,train_data_path,test_data_path):
    # define the transformations for the images
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load dataset
    trainset = ImageFolder(train_data_path,transform=transform_train)
    testset = ImageFolder(test_data_path,transform=transform_test)

    # split the train_set into train and validation sets
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
