
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

from utils.utils import save_model, Struct, set_seed, Wandb_Writer
from datasets.datasets_fft import build_HDF5_feat_dataset
from architecture_fft.ibmil import IBMIL_fft

import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.special import softmax
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
import numpy as np

from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from timm.utils import accuracy
import torchmetrics
import wandb

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def get_arguments():
    parser = argparse.ArgumentParser('WSI classification training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/luad_fft_config.yml',
                        help='settings of Tip-Adapter in yaml format')
    parser.add_argument(
        "--eval-only", action="store_true", help="evaluation only"
    )
    parser.add_argument(
        "--seed", type=int, default=5, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--wandb_mode', default='disabled', choices=['offline', 'online', 'disabled'],
                        help='the model of wandb')
    parser.add_argument('--c_path', action='store_true', help='directory to confounders')
    parser.add_argument('--c_learn', action='store_true', help='learn confounder or not')

    parser.add_argument('--pretrain', default='natural_supervised',
                        choices=['natural_supervised', 'medical_ssl', 'path-clip-L-336'],
                        help='settings of Tip-Adapter in yaml format')
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


    conf.D_feat = 1024
    conf.D_inner = 256

    if conf.c_path:
        conf.c_path = ['./datasets_deconf/%s/train_bag_cls_agnostic_feats_proto_8_pretrain_%s_seed_%s.npy'%(conf.dataset, conf.pretrain, conf.seed)]


    # Prepare dataset
    set_seed(args.seed)

    # define datasets and dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset(os.path.join(conf.data_dir, 'patch_feats_pretrain_%s.h5'%conf.pretrain), conf, args.quick)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    # define network
    model = IBMIL_fft(conf)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # define optimizer, lr not important at this point
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=conf.wd)

    import math
    best_f1, best_roc_auc, best_loss, flag = 0, 0, math.inf, False

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    for epoch in range(conf.train_epoch):

        train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, conf)

        # val_auc, val_acc, val_f1, val_loss = evaluate(net, criterion, val_loader, device, conf, 'Val')
        test_loss_bag, metrics = evaluate(model, criterion, test_loader, device, conf, 'Test')
        accuracy, precision_macro, recall_macro, f1_macro, report, conf_matrix, auc_roc, auc_roc_per_class = metrics
        save_dir = './results/ibmil_LUAD_fft_add_lvl1/'
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

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, conf):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100


    for data_it, data in enumerate(data_loader):
        # for data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        # Move input batch onto GPU if eager execution is enabled (default), else leave it on CPU
        # Data is a dict with keys `input` (patches) and `{task_name}` (labels for given task)
        fft = torch.tensor(np.load(data['fft'][0])['arr_0']).unsqueeze(0).to(device)
        image_patches = torch.load(data['input'][0], weights_only=True).to(device, dtype=torch.float32).unsqueeze(0)
        labels = data['label'].to(device)

        # # Calculate and set new learning rate
        adjust_learning_rate(optimizer, epoch + data_it/len(data_loader), conf)
        # adjust_learning_rate(optimizer1, epoch + data_it/len(data_loader), conf)

        # Compute loss
        preds, feats, attn = model(image_patches, fft)



        loss = criterion(preds, labels)

        optimizer.zero_grad()
        # Backpropagate error and update parameters
        loss.backward()
        optimizer.step()


        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(loss=loss.item())



        # if conf.wandb_mode != 'disabled':
        #     """ We use epoch_1000x as the x-axis in tensorboard.
        #     This calibrates different curves when batch size changes.
        #     """
        #     wandb.log({'loss': loss})



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
        fft = torch.tensor(np.load(data['fft'][0])['arr_0']).unsqueeze(0).to(device)
        image_patches = torch.load(data['input'][0], weights_only=True).to(device, dtype=torch.float32).unsqueeze(0)
        labels = data['label'].to(device)


        logits, feats, attn = net(image_patches, fft)
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

