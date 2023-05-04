import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import data
import seaborn as sns
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_loss_accuracy(loss_list, val_loss_list, test_loss_list, acc_list, test_acc_list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(range(len(loss_list)), loss_list, label='running_loss')
    ax1.plot(range(len(val_loss_list)), val_loss_list, label='val_loss')
    ax1.plot(range(len(test_loss_list)), test_loss_list, label='test_loss')
    ax1.set_title('Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(range(len(acc_list)), acc_list, label='val_acc')
    ax2.plot(range(len(test_acc_list)), test_acc_list, label='test_acc')
    ax2.set_title('Train Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy(%)')
    ax2.legend()

    plt.savefig(f'F:\\TEST\\ture_fig_loss_acc_1.png')


def plot_recall(precision_list, recall_list, f1_list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(range(len(precision_list)), precision_list, label='precision')
    ax1.plot(range(len(recall_list)), recall_list, label='recall')
    ax1.plot(range(len(f1_list)), f1_list, label='f1')
    ax1.set_title('RECALL')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('%')
    ax1.legend()

    plt.savefig(f'F:\\TEST\\ture_recall_1.png')


def save_metrics_to_txt(loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list, txt_file_path):
    file_names = ["loss_list.txt", "val_loss_list.txt", "test_loss_list.txt", "val_acc_list.txt", "test_acc_list.txt"]
    lists = [loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list]

    for file_name, data_list in zip(file_names, lists):
        with open(os.path.join(txt_file_path, file_name), 'w') as f:
            f.write("Epoch, Value\n")
            for i, value in enumerate(data_list):
                f.write(f"{i + 1}, {value}\n")


def save_recall_to_txt(precision_list, recall_list, f1_list, txt_file_path):
    file_names = ["precision_list.txt", "recall_list.txt", "f1.txt"]
    lists = [precision_list, recall_list, f1_list]

    for file_name, data_list in zip(file_names, lists):
        with open(os.path.join(txt_file_path, file_name), 'w') as f:
            f.write("Epoch, Value\n")
            for i, value in enumerate(data_list):
                f.write(f"{i + 1}, {value}\n")


def evaluate(model, criterion, dataload_test):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    with torch.no_grad():
        for imgs, labels in dataload_test:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_num += imgs.size(0)

    avg_loss = total_loss / total_num
    accuracy = total_correct / total_num
    return avg_loss, accuracy


def evaluate_recall(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            y_true.extend(targets.cpu().numpy().tolist())

    precision = precision_score(y_true, y_pred, average=None)  # 适用于多分类
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    return precision, recall, f1, y_pred, y_true


def plot_confusion_matrix_heatmap(true_labels, pred_labels, class_names):  # 绘制热力图
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix Heatmap')

    plt.show()
    plt.savefig(f'F:\\TEST\\heatmap_1.png')


def train_model(model, criterion, optimizer, dataload_train, dataload_test, test_loader, num_epochs, batch_size,
                weights_path, txt_file_path):
    if os.path.isfile(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path))
            print('successful load weight！')
        except Exception as e:
            print(f'Error loading weights: {e}')
    else:
        print('Weight file does not exist, training from scratch.', )


    loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list = [], [], [], [], []
    precision_list, recall_list, f1_list = [], [], []
    class_names = [' 1', '2', '3', '4', '5','6','7']  # 多分类
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
                running_loss = criterion(outputs, labels)
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

            val_loss, val_accuracy = evaluate(model, criterion, dataload_test)
            print('val loss: {:.6f}'.format(val_loss))
            print('Validation accuracy: {:.6f}%'.format(val_accuracy * 100))
            val_loss_list.append(val_loss)
            val_acc_list.append(val_accuracy * 100)

            test_loss, test_accuracy = evaluate(model, criterion, test_loader)
            print('test loss: {:.6f}'.format(test_loss))
            print('Test accuracy: {:.6f}%'.format(test_accuracy * 100))
            test_loss_list.append(test_loss)
            test_acc_list.append(test_accuracy * 100)

            precision, recall, f1, y_pred, y_true = evaluate_recall(model, test_loader, device)
            print('precision',precision)
            print('recall',recall)
            print('f1',f1)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        plot_loss_accuracy(loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list)

        plot_recall(precision_list, recall_list, f1_list)

        plot_confusion_matrix_heatmap(y_true, y_pred, class_names=class_names)

        save_metrics_to_txt(loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list,
                            txt_file_path=txt_file_path)

        save_recall_to_txt(precision_list, recall_list, f1_list, txt_file_path=txt_file_path)
        # save_metrics_to_txt()
        print(f'acc and loss path at {txt_file_path}')
    return model, loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list


# train
def train(model, criterion, optimizer, train_data_path, test_data_path, weights_path, batch_size, num_epochs, file_path,
          num_workers):
    print(f"Weight path: {weights_path}")
    trainloader, valloader, testloader = data.load_data(batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        train_data_path=train_data_path,
                                                        test_data_path=test_data_path)

    model, loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list = train_model(model,
                                                                                               criterion,
                                                                                               optimizer,
                                                                                               trainloader,
                                                                                               valloader,
                                                                                               testloader,
                                                                                               num_epochs,
                                                                                               batch_size,
                                                                                               weights_path,
                                                                                               file_path)
    print("Finished training")
    return model, loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list
