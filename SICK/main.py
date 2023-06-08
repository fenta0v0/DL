import argparse
import torch

import mobileViT
import Fnet
import resnet  # load net
from utils_test import train  # 多分类
#from utils import train  # 二分类
from data import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification Training")
    parser.add_argument("--train_data_path", type=str, default=r"F:\AutoDL\DATA_ture\ture\01\train",
                        help="Path to the training data")
    parser.add_argument("--test_data_path", type=str, default=r"F:\AutoDL\DATA_ture\ture\01\test",
                        help="Path to the test data")
    parser.add_argument("--weights_path", type=str, default=r'F:\TEST\zong\test\mobileVIT_1_3_ture.pth',
                        help="Path to the model's weights")  # 文件命名方式  model name__num_class__test or formal(0or1)__num
    parser.add_argument('--file_path', type=str, default=r"F:\TEST\zong\test\txt",
                        help="save txt file path")
    parser.add_argument('--png_save_path', type=str, default=r"F:\TEST\zong\test\png",
                        help="save txt file path")
    parser.add_argument('--roc_save_path', type=str, default=r"F:\TEST\zong\test\png",
                        help="save txt file path")

    parser.add_argument("--batch_size", type=int, default=6,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of epochs for training")
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of Thread Workings')

    args = parser.parse_args()

    # load net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance = mobileViT.mobile_vit_small(2)
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
        file_path=args.file_path,
        num_workers=args.num_workers,
        png_path=args.png_save_path,
        roc_save_path=args.roc_save_path
    )
