"""
二分类任务
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, matthews_corrcoef, \
    average_precision_score,precision_recall_curve
import data
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy import interp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_loss_accuracy(loss_list, val_loss_list, test_loss_list, acc_list, test_acc_list, save_path):
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

    plt.savefig(os.path.join(save_path, 'loss_acc.png'))


def plot_recall(precision_list, recall_list, f1_list, mcc_list, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(range(len(precision_list)), precision_list, label='precision')
    ax1.plot(range(len(recall_list)), recall_list, label='recall')
    ax1.plot(range(len(f1_list)), f1_list, label='f1')
    ax1.set_title('RECALL')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('%')
    ax1.legend()

    ax2.plot(range(len(mcc_list)), mcc_list, label='MCC')
    ax2.set_title('MCC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('%')
    ax2.legend()

    plt.savefig(os.path.join(save_path, 'recall.png'))

def plot_roc_curve(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, 'ROC.png'))
    plt.show()


def plot_precision_recall_curve(y_true, y_scores, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {average_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(save_path, 'PR.png'))
    plt.show()


# 保存loss和acc
def save_metrics_to_txt(loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list, txt_file_path):
    file_names = ["loss_list.txt", "val_loss_list.txt", "test_loss_list.txt", "val_acc_list.txt", "test_acc_list.txt"]
    lists = [loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list]

    for file_name, data_list in zip(file_names, lists):
        with open(os.path.join(txt_file_path, file_name), 'w') as f:
            f.write("Epoch, Value\n")
            for i, value in enumerate(data_list):
                f.write(f"{i + 1}, {value}\n")


def save_recall_to_txt(precision_list, recall_list, f1_list, mcc_list, txt_file_path):
    file_names = ["precision_list.txt", "recall_list.txt", "f1.txt", "mcc.txt"]
    lists = [precision_list, recall_list, f1_list, mcc_list]

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
    y_score = []  # 添加一个列表来保存预测概率

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            y_true.extend(targets.cpu().numpy().tolist())

            # 计算预测概率并保存
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            y_score.extend(probabilities[:, 1].cpu().numpy().tolist())

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    return precision, recall, f1, y_pred, y_true, y_score, mcc  # 修改返回值，增加y_score


def plot_confusion_matrix_heatmap(true_labels, pred_labels, class_names, save_path):  # 绘制热力图
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix Heatmap')

    plt.savefig(os.path.join(save_path, 'heatmap.png'))
    plt.show()


def train_model(model, criterion, optimizer, dataload_train, dataload_test, test_loader, num_epochs, batch_size,
                weights_path, txt_file_path, png_path, roc_save_path):
    if os.path.isfile(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path))
            print('successful load weight！')
        except Exception as e:
            print(f'Error loading weights: {e}')
    else:
        print('Weight file does not exist, training from scratch.', )

    best_accuracy = 0.0
    best_epoch = -1
    loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list = [], [], [], [], []
    precision_list, recall_list, f1_list, mcc_list = [], [], [], []
    class_names = [' 0', '1']  # 二分类

    roc_results = []
    pr_results = []

    # Define a common grid for all ROC curves
    base_fpr = np.linspace(0, 1, 101)

    # Define a common grid for all PR curves
    base_recall = np.linspace(0, 1, 101)

    mean_tpr = 0.0
    mean_precision = 0.0
    aucs = []
    tprs = []
    precisions = []
    pr_aucs = []

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

            # precision, recall, f1, y_pred, y_true = evaluate_recall(model, test_loader, device)
            precision, recall, f1, y_pred, y_true, y_score, mcc = evaluate_recall(model, test_loader, device)
            print('precision', precision)
            print('recall', recall)
            print('f1', f1)
            print('mcc', mcc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            mcc_list.append(mcc)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = epoch
                torch.save(model.state_dict(), weights_path)
                print(f"Best model weights saved at {weights_path}")

        plot_loss_accuracy(loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list, png_path)

        plot_recall(precision_list, recall_list, f1_list, mcc_list, png_path)

        plot_confusion_matrix_heatmap(y_true, y_pred, save_path=png_path, class_names=class_names)

        plot_roc_curve(y_true, y_score, roc_save_path)

        plot_precision_recall_curve(y_true,y_score,roc_save_path)

        save_metrics_to_txt(loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list,
                            txt_file_path=txt_file_path)

        save_recall_to_txt(precision_list, recall_list, f1_list, mcc_list, txt_file_path=txt_file_path)
        # save_metrics_to_txt()
        print(f'acc and loss path at {txt_file_path}')

        ###############extra############

        # Compute mean and standard deviation of AUC
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Interpolate tpr
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0

        # Compute PR curve and PR area
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)

        # Interpolate precision
        precision = interp(base_recall, recall[::-1], precision[::-1])

        tprs.append(tpr)
        precisions.append(precision)
        aucs.append(roc_auc)
        pr_aucs.append(pr_auc)

        # Keep only the last 10 results
        tprs = tprs[-10:]
        precisions = precisions[-10:]
        aucs = aucs[-10:]

        # Compute mean and standard deviation of tpr and precision
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_precision = np.mean(precisions, axis=0)
        std_precision = np.std(precisions, axis=0)
        mean_pr_auc = np.mean(pr_aucs)
        std_pr_auc = np.std(pr_aucs)

        # Compute confidence interval
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        precisions_upper = np.minimum(mean_precision + std_precision, 1)
        precisions_lower = np.maximum(mean_precision - std_precision, 0)

        # Plot ROC Curve
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        #plt.plot(base_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.4f)' % aucs[-1])
        plt.plot(base_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc))
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='green', alpha=.2, label=r'$\pm$ 1 std. dev.')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)  # 添加45度线
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        # Plot PR Curve
        plt.subplot(1, 2, 2)
        #plt.plot(base_recall, mean_precision, color='r', label=r'Mean PR (AUC = %0.4f)' % pr_auc)
        plt.plot(base_recall, mean_precision, color='r', label=r'Mean PR (AUC = %0.4f $\pm$ %0.4f)' % (mean_pr_auc, std_pr_auc))
        plt.fill_between(base_recall, precisions_lower, precisions_upper, color='pink', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(png_path,'ROC_PR.png'))
        plt.show()

    return model, loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list

    print("Training completed.")
    print(f"Best accuracy achieved: {best_accuracy:.6f}% at epoch {best_epoch + 1}")


# train
def train(model, criterion, optimizer, train_data_path, test_data_path, weights_path, batch_size, num_epochs, file_path,
          num_workers, png_path, roc_save_path):
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
                                                                                               file_path,
                                                                                               png_path,
                                                                                               roc_save_path)
    print("Finished training")
    return model, loss_list, val_loss_list, test_loss_list, val_acc_list, test_acc_list
