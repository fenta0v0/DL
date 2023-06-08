import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import data
import itertools
from skimage import measure
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def analyze_image(images):
    if isinstance(images, torch.Tensor):
        if images.is_cuda:
            images = images.cpu()
        images = images.detach().numpy()

    # 存储每一张图像的块信息
    batch_blocks_info = []

    for img in images:
        img = img.squeeze()  # 移除大小为1的维度

        # 将 img 的值范围映射到 [0, 255] 并转换类型为 uint8
        img = ((img - img.min()) * (255.0 / (img.max() - img.min()))).astype(np.uint8)

        if img.ndim > 2:  # 如果图像有多个通道
            img = img[:, :, 0]  # 只取第一个通道进行处理

        _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blocks_info = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w / 2
            center_y = y + h / 2

            mask = np.zeros(thresh.shape, np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            mean_val = cv2.mean(img, mask=mask)

            block_info = {
                "center": (center_x, center_y),
                "width": w,
                "height": h,
                "mean_gray_value": mean_val[0],
            }

            blocks_info.append(block_info)

            # 打印每个块的检测结果
            print("Detected block info: center=({}, {}), width={}, height={}, mean_gray_value={}"
                  .format(center_x, center_y, w, h, mean_val[0]))

        # 添加当前图像的块信息到批量的块信息中
        batch_blocks_info.append(blocks_info)

    return batch_blocks_info


def loss_fn(preds, labels):
    count_loss = 0
    center_loss = 0
    size_loss = 0
    total_blocks = 0

    for pred, label in zip(preds, labels):
        pred_blocks_info = analyze_image(pred)
        label_blocks_info = analyze_image(label)

        count_loss += F.mse_loss(torch.tensor([len(pred_blocks_info)], dtype=torch.float32),
                                 torch.tensor([len(label_blocks_info)], dtype=torch.float32))

        total_blocks += min(len(pred_blocks_info), len(label_blocks_info))

        for pred_info, label_info in zip(pred_blocks_info, label_blocks_info):
            pred_center = torch.tensor(pred_info["center"], dtype=torch.float32)
            label_center = torch.tensor(label_info["center"], dtype=torch.float32)
            center_loss += F.mse_loss(pred_center, label_center)

            pred_size = torch.tensor([pred_info["width"], pred_info["height"]], dtype=torch.float32)
            label_size = torch.tensor([label_info["width"], label_info["height"]], dtype=torch.float32)
            size_loss += F.mse_loss(pred_size, label_size)

    return count_loss / len(preds), center_loss / total_blocks, size_loss / total_blocks



def evaluate(model, criterion, dataloader, epoch):
    model.eval()
    total_loss, total_count_loss, total_center_loss, total_size_loss = 0, 0, 0, 0
    total_samples = 0
    ssim_total, psnr_total = 0, 0

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = model(imgs)
            count_loss, center_loss, size_loss = loss_fn(pred, labels)

            total_loss += count_loss.item() + center_loss.item() + size_loss.item()
            total_count_loss += count_loss.item()
            total_center_loss += center_loss.item()
            total_size_loss += size_loss.item()
            total_samples += imgs.size(0)

            for pred_single, label_single in zip(pred, labels):
                pred_single = pred_single.cpu().numpy()
                label_single = label_single.cpu().numpy()

                win_size = min(pred_single.shape[0], pred_single.shape[1], label_single.shape[0], label_single.shape[1])

                # Make sure win_size is odd
                if win_size % 2 == 0:
                    win_size -= 1

                try:
                    ssim_total += ssim(pred_single, label_single, win_size=win_size, data_range=1, multichannel=True)
                except ZeroDivisionError:
                    ssim_total += 0  # 或者你想要处理这个错误的其他方式
                psnr_total += psnr(label_single, pred_single, data_range=1)

            # 随机选取一个batch进行保存
            if i == np.random.randint(len(dataloader)):
                # 将预测结果拼接到原始图片上
                imgs = imgs.cpu()
                pred = pred.cpu()
                result = torch.cat((imgs, pred), dim=2) # 按照宽度拼接
                save_image(result, f'F:/Code_1/results/result_epoch_{epoch}.png')

    # compute average losses and metrics
    avg_loss = total_loss / total_samples
    avg_count_loss = total_count_loss / total_samples
    avg_center_loss = total_center_loss / total_samples
    avg_size_loss = total_size_loss / total_samples
    avg_ssim = ssim_total / total_samples
    avg_psnr = psnr_total / total_samples

    print('Average Loss: {:.4f}, Average Count Loss: {:.4f}, Average Center Loss: {:.4f}, Average Size Loss: {:.4f}, Average SSIM: {:.4f}, Average PSNR: {:.4f}'.format(avg_loss, avg_count_loss, avg_center_loss, avg_size_loss, avg_ssim, avg_psnr))

    return avg_loss, avg_count_loss, avg_center_loss, avg_size_loss, avg_ssim, avg_psnr



def plot_loss_png(test_loss_list, test_count_loss_list, test_center_loss_list, test_size_loss_list, test_ssim_list, test_psnr_list, save_path):
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))

    ax[0, 0].plot(range(len(test_loss_list)), test_loss_list, label='Test SIMM Loss')
    ax[0, 0].set_title('Train SIMM Loss')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].legend()

    ax[0, 1].plot(range(len(test_count_loss_list)), test_count_loss_list, label='Test Count Loss')
    ax[0, 1].set_title('Test Count Loss')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_ylabel('Loss')
    ax[0, 1].legend()

    ax[1, 0].plot(range(len(test_center_loss_list)), test_center_loss_list, label='Test Center Loss')
    ax[1, 0].set_title('Test Center Loss')
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].set_ylabel('Loss')
    ax[1, 0].legend()

    ax[1, 1].plot(range(len(test_size_loss_list)), test_size_loss_list, label='Test Size Loss')
    ax[1, 1].set_title('Test Size Loss')
    ax[1, 1].set_xlabel('Epoch')
    ax[1, 1].set_ylabel('Loss')
    ax[1, 1].legend()

    # Add SSIM plot
    ax[2, 0].plot(range(len(test_ssim_list)), test_ssim_list, label='Test SSIM')
    ax[2, 0].set_title('Test SSIM')
    ax[2, 0].set_xlabel('Epoch')
    ax[2, 0].set_ylabel('SSIM')
    ax[2, 0].legend()

    # Add PSNR plot
    ax[2, 1].plot(range(len(test_psnr_list)), test_psnr_list, label='Test PSNR')
    ax[2, 1].set_title('Test PSNR')
    ax[2, 1].set_xlabel('Epoch')
    ax[2, 1].set_ylabel('PSNR')
    ax[2, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_metrics.png'))
    plt.show()



def save_metrics_to_txt(test_loss_list, test_count_loss_list, test_center_loss_list, test_size_loss_list,
                        txt_file_path):
    file_names = ["test_loss_list.txt", "test_count_loss_list.txt", "test_center_loss_list.txt",
                  "test_size_loss_list.txt"]
    lists = [test_loss_list, test_count_loss_list, test_center_loss_list, test_size_loss_list]

    for file_name, data_list in zip(file_names, lists):
        with open(os.path.join(txt_file_path, file_name), 'w') as f:
            f.write("Epoch, Value\n")
            for i, value in enumerate(data_list):
                f.write(f"{i + 1}, {value}\n")


def save_metrics_to_txt_train(train_loss_list, txt_file_path):
    file_name = "train_loss_list.txt"
    with open(os.path.join(txt_file_path, file_name), 'w') as f:
        f.write("Epoch, Value\n")
        for i, value in enumerate(train_loss_list):
            f.write(f"{i + 1}, {value}\n")


def plot_loss_train(train_loss_list, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'train_loss.png'))


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(output, dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def train_model(model, criterion, optimizer, dataload_train, dataload_test, num_epochs, batch_size,
                weights_path, txt_file_path, png_path):
    if os.path.isfile(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path))
            print('successful load weight！')
        except Exception as e:
            print(f'Error loading weights: {e}')
    else:
        print('Weight file does not exist, training from scratch.', )

    loss_list, test_loss_list, test_count_loss, test_center_loss, test_size_loss,test_ssim,test_psnr = [], [], [], [], [],[],[]

    for epoch in range(num_epochs):
        dataset_size = len(dataload_train.dataset)
        epoch_loss = 0
        step = 0
        with tqdm(total=dataset_size, desc=f'Epoch [{epoch + 1}/{num_epochs}]') as pbar:
            for imgs, labels in dataload_train:
                optimizer.zero_grad()
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                running_loss = criterion(labels, outputs)
                running_loss.backward()
                optimizer.step()
                epoch_loss += running_loss.item()

                pbar.set_postfix(mini_loss=epoch_loss / (step + 1), epoch_loss=float(epoch_loss),
                                 running_loss=float(running_loss))
                pbar.update(batch_size)
                step += 1

            loss_list.append(epoch_loss / len(dataload_train))

            try:
                torch.save(model.state_dict(), weights_path)
                print(f"Model saved successfully at {weights_path}")
            except Exception as e:
                print(f"Error saving the model at {weights_path}: {e}")

            plot_loss_train(loss_list, save_path=png_path)
            save_metrics_to_txt_train(loss_list, txt_file_path=txt_file_path)

            # TEST
            avg_loss, avg_count_loss, avg_center_loss, avg_size_loss,ssim,psnr = evaluate(model, criterion, dataload_test,epoch)

            test_loss_list.append(avg_loss)
            test_count_loss.append(avg_count_loss)
            test_center_loss.append(avg_center_loss)
            test_size_loss.append(avg_size_loss)
            test_ssim.append(ssim)
            test_psnr.append(psnr)

        plot_loss_png(test_loss_list, test_count_loss, test_center_loss, test_size_loss,test_ssim,test_psnr, png_path)

        save_metrics_to_txt(test_loss_list, test_count_loss, test_center_loss, test_size_loss,
                            txt_file_path=txt_file_path)

        # save_metrics_to_txt()
        print(f'acc and loss path at {txt_file_path}')
    return model
    print("Training completed.")
    print(f"Best accuracy achieved: {best_accuracy:.6f}% at epoch {best_epoch + 1}")


# train
def train(model, criterion, optimizer, train_data_path, test_data_path, weights_path, batch_size, num_epochs, file_path,
          num_workers, png_path):
    print(f"Weight path: {weights_path}")
    trainloader, testloader = data.load_data(batch_size=batch_size,
                                             num_workers=num_workers,
                                             train_data_path=train_data_path,
                                             test_data_path=test_data_path)

    model = train_model(model,
                        criterion,
                        optimizer,
                        trainloader,
                        testloader,
                        num_epochs,
                        batch_size,
                        weights_path,
                        file_path,
                        png_path
                        )
    print("Finished training")
    return model
