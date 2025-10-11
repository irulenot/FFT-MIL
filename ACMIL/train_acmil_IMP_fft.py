# !/usr/bin/env python
import sys
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
from pprint import pprint

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import save_model, Struct, set_seed
from datasets.datasets_fft import build_HDF5_feat_dataset
from architecture_fft.transformer import ACMIL_GA_fft

import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.special import softmax
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
import numpy as np

from timm.utils import accuracy
import torchmetrics
import wandb

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def get_arguments():
    parser = argparse.ArgumentParser('WSI classification training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/imp_config.yml',
                        help='settings of dataset in yaml format')
    parser.add_argument(
        "--eval-only", action="store_true", help="evaluation only"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--wandb_mode', default='disabled', choices=['offline', 'online', 'disabled'],
                        help='the model of wandb')
    parser.add_argument(
        "--n_token", type=int, default=5, help="number of attention branches in (MBA)."
    )
    parser.add_argument(
        "--n_masked_patch", type=int, default=10, help="top-K instances are be randomly masked in STKIM."
    )
    parser.add_argument(
        "--mask_drop", type=float, default=0.6, help="maksing ratio in the STKIM"
    )
    parser.add_argument("--arch", type=str, default='ga', choices=['ga', 'mha'], help="choice of architecture type")
    parser.add_argument('--pretrain', default='natural_supervised',
                        choices=['natural_supervised', 'medical_ssl', 'plip', 'path-clip-B-AAAI'
                                 'openai-clip-B', 'openai-clip-L-336', 'quilt-net', 'path-clip-B', 'path-clip-L-336',
                                 'biomedclip', 'path-clip-L-768', 'UNI', 'GigaPath'],
                        help='pretrained backbone')
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument(
        "--quick", default=False
    )
    args = parser.parse_args()
    return args

def main():
    # Load config file
    args = get_arguments()

    # get config
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    if conf.pretrain == 'medical_ssl':
        conf.D_feat = 384
        conf.D_inner = 128
    elif conf.pretrain == 'natural_supervised':
        conf.D_feat = 512
        conf.D_inner = 256
    elif conf.pretrain == 'path-clip-B' or conf.pretrain == 'openai-clip-B' or conf.pretrain == 'plip'\
            or conf.pretrain == 'quilt-net'  or conf.pretrain == 'path-clip-B-AAAI'  or conf.pretrain == 'biomedclip':
        conf.D_feat = 512
        conf.D_inner = 256
    elif conf.pretrain == 'path-clip-L-336' or conf.pretrain == 'openai-clip-L-336':
        conf.D_feat = 768
        conf.D_inner = 384
    elif conf.pretrain == 'UNI':
        conf.D_feat = 1024
        conf.D_inner = 512
    elif conf.pretrain == 'GigaPath':
        conf.D_feat = 1536
        conf.D_inner = 768

    # Prepare dataset
    set_seed(args.seed)

    # define datasets and dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset(os.path.join(conf.data_dir, 'tumor_vs_normal_resnet_features.h5'), conf, args.quick)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    # define network
    if conf.arch == 'ga':
        conf.D_feat = 1024
        model = ACMIL_GA_fft(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # define optimizer, lr not important at this point
    optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=conf.wd)

    import math
    best_f1, best_roc_auc, best_loss, flag = 0, 0, math.inf, False

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    for epoch in range(conf.train_epoch):

        train_one_epoch(model, criterion, train_loader, optimizer0, device, epoch, conf)

        # val_auc, val_acc, val_f1, val_loss = evaluate(net, criterion, val_loader, device, conf, 'Val')
        test_loss_bag, metrics = evaluate(model, criterion, test_loader, device, conf, 'Test')
        accuracy, precision_macro, recall_macro, f1_macro, report, conf_matrix, auc_roc, auc_roc_per_class = metrics
        save_dir = './results/acmil_IMP_fft_add2/'
        os.makedirs(save_dir, exist_ok=True)
        if f1_macro > best_f1:
            print(f"F1: best loss: {test_loss_bag:.4f}, best f1: {f1_macro:.4f}, best acc: {accuracy:.4f}, best roc_auc: {auc_roc:.4f}")
            np.save(os.path.join(save_dir, 'metrics_f1.npy'), np.array(metrics, dtype=object))
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_f1.pt'))
            best_f1 = f1_macro
            flag = True
        if auc_roc > best_roc_auc:
            print(f"AUC_ROC: best loss: {test_loss_bag:.4f}, best f1: {f1_macro:.4f}, best acc: {accuracy:.4f}, best roc_auc: {auc_roc:.4f}")
            np.save(os.path.join(save_dir, 'metrics_auc_roc.npy'), np.array(metrics, dtype=object))
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_auc_roc.pt'))
            best_roc_auc = auc_roc
            flag = True
        if test_loss_bag < best_loss:
            print(f"LOSS: best loss: {test_loss_bag:.4f}, best f1: {f1_macro:.4f}, best acc: {accuracy:.4f}, best roc_auc: {auc_roc:.4f}")
            np.save(os.path.join(save_dir, 'metrics_loss.npy'), np.array(metrics, dtype=object))
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_loss.pt'))
            best_loss = test_loss_bag
            flag = True
        if flag:
            print()
            flag = False

    print(f"Best F1: {best_f1:.4f}, Best AUC_ROC: {best_roc_auc:.4f}, Best Loss: {best_loss:.4f}")
    return None

def train_one_epoch(model, criterion, data_loader, optimizer0, device, epoch, conf):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    model.train()

    for data_it, data in enumerate(data_loader):
        # for data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        # Move input batch onto GPU if eager execution is enabled (default), else leave it on CPU
        # Data is a dict with keys `input` (patches) and `{task_name}` (labels for given task)
        image_patches = torch.load(data['input'][0], weights_only=True).to(device, dtype=torch.float32).unsqueeze(0)
        fft =  torch.tensor(np.load(data['fft'][0])['arr_0']).unsqueeze(0).to(device)
        labels = data['label'].to(device)

        # # Calculate and set new learning rate
        adjust_learning_rate(optimizer0, epoch + data_it/len(data_loader), conf)

        # Compute loss
        sub_preds, slide_preds, attn = model(image_patches, fft)
        if conf.n_token > 1:
            loss0 = criterion(sub_preds, labels.repeat_interleave(conf.n_token))
        else:
            loss0 = torch.tensor(0.)
        loss1 = criterion(slide_preds, labels)


        diff_loss = torch.tensor(0, device=device, dtype=torch.float)
        attn = torch.softmax(attn, dim=-1)

        for i in range(conf.n_token):
            for j in range(i + 1, conf.n_token):
                diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                            conf.n_token * (conf.n_token - 1) / 2)

        loss = diff_loss + loss0 + loss1

        optimizer0.zero_grad()
        # Backpropagate error and update parameters
        loss.backward()
        optimizer0.step()

        if conf.wandb_mode != 'disabled':
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            wandb.log({'sub_loss': loss0}, commit=False)
            wandb.log({'diff_loss': diff_loss}, commit=False)
            wandb.log({'slide_loss': loss1})




from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize
# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(net, criterion, data_loader, device, conf, header):

    # Set the network to evaluation mode
    net.eval()

    val_loss = 0
    test_labels = []
    test_predictions = []

    metric_logger = MetricLogger(delimiter="  ")

    for data in data_loader:
        image_patches = torch.load(data['input'][0], weights_only=True).to(device, dtype=torch.float32).unsqueeze(0)
        fft =  torch.tensor(np.load(data['fft'][0])['arr_0']).unsqueeze(0).to(device)
        labels = data['label'].to(device)

        sub_preds, logits, attn = net(image_patches, fft)
        div_loss = torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1)) / attn.shape[1]
        loss = criterion(logits, labels)
        val_loss += loss.item()

        label = F.one_hot(labels.to('cpu'), num_classes=len(logits[0]))
        test_labels.extend([label.squeeze().cpu().numpy()])
        Y_prob = F.softmax(logits, dim = 1)
        test_predictions.extend([Y_prob.squeeze().cpu().numpy()])

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)    
    test_labels_class = np.argmax(test_labels, axis=1)
    test_predictions_class = np.argmax(test_predictions, axis=1)

     # Basic metrics
    accuracy = accuracy_score(test_labels_class, test_predictions_class)
    precision_macro = precision_score(test_labels_class, test_predictions_class, average='macro', zero_division=0)
    recall_macro = recall_score(test_labels_class, test_predictions_class, average='macro', zero_division=0)
    f1_macro = f1_score(test_labels_class, test_predictions_class, average='macro', zero_division=0)

    # Classification report (per-class metrics)
    report = classification_report(test_labels_class, test_predictions_class, output_dict=True, zero_division=0)

    # Confusion matrix
    conf_matrix = confusion_matrix(test_labels_class, test_predictions_class)

    # AUC-ROC (requires probability scores or binarized labels)
    # If predictions are labels only, binarize for ROC-AUC
    classes = np.unique(test_labels_class)
    y_true_bin = label_binarize(test_labels_class, classes=classes)
    if len(classes) == 2:
        y_true_bin = np.eye(len(classes))[test_labels_class]
    else: 
        y_true_bin = label_binarize(test_labels_class, classes=classes)
        
    # Multi-class AUC-ROC using macro averaging
    auc_roc = roc_auc_score(y_true_bin, test_predictions, average='macro', multi_class='ovr')
    auc_roc_per_class = roc_auc_score(y_true_bin, test_predictions, average=None, multi_class='ovr')  # shape: (num_classes,)

    metrics = [accuracy, precision_macro, recall_macro, f1_macro, report, conf_matrix, auc_roc, auc_roc_per_class]
    return val_loss / len(data_loader), metrics

if __name__ == '__main__':
    main()

