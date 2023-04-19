import argparse
import mobileVIT
import torch

import resnet
from utils import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileViT Image Classification Training")
    parser.add_argument("--train_data_path", type=str, default=r"E:\sick_datasets\z\train", help="Path to the training data")
    parser.add_argument("--test_data_path", type=str, default=r"E:\sick_datasets\z\test", help="Path to the test data")
    parser.add_argument("--weights_path", type=str, default=r"E:\test_code\mobileVIT\params\resnet_1_2_20.pth", help="Path to the model's weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument('--file_path',type=str, default=r"E:\test_code\mobileVIT\txt\1.txt", help="save txt file path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance = resnet.resnet50(2)
    model_instance = model_instance.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_instance.parameters(), lr=1e-4, momentum=0.9)

    model, loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list = train(
        model=model_instance,
        criterion=criterion,
        optimizer=optimizer,
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        weights_path=args.weights_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        file_path=args.file_path
    )


