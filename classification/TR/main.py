import argparse
import torch

import mobileViT
import resnet  # load net
from utils1 import train  # load Training Model  多分类
# from utils import train  # 二分类
from data import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification Training")
    parser.add_argument("--train_data_path", type=str, default=r"E:\sick_classfication\ture\zz_train",
                        help="Path to the training data")
    parser.add_argument("--test_data_path", type=str, default=r"E:\sick_classfication\ture\zz_test",
                        help="Path to the test data")
    parser.add_argument("--weights_path", type=str, default=r'F:/TEST/params_ture/mobilevit_ture_224_7_1_1.pth',
                       help="Path to the model's weights")  # 文件命名方式  model name__num_class__test or formal(0or1)__num
    parser.add_argument('--file_path', type=str, default=r"F:/TEST/txt_class/demo/224_7",
                        help="save txt file path")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of epochs for training")
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of Thread Workings')
    args = parser.parse_args()

    # load net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance = mobileViT.mobile_vit_small(7)
    model_instance = model_instance.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_instance.parameters(), lr=1e-4, momentum=0.9)
    model, loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list = train(
        model=model_instance,
        criterion=criterion,
        optimizer=optimizer,
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        weights_path= args.weights_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        file_path=args.file_path,
        num_workers=args.num_workers
    )


