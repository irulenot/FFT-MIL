import sys
import numpy as np
import torch
import time
import torch.nn.functional as F

from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate, adjust_learning_rate_StepLR
from timm.utils import accuracy
import torchmetrics
from architecture.bmil import get_ard_reg_vdo

import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.special import softmax
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize

def loss_forward_and_backward(net, image_patches, fft, labels, criterion, conf,
                              device, optimizer, metric_logger, log_writer=None):
    # Compute loss
    preds = net(image_patches, fft)
    ce_loss = criterion(preds, labels)

    diff_loss = torch.tensor(0).to(device, dtype=torch.float)

    loss = conf.w_loss * diff_loss + ce_loss

    # Backpropagate error and update parameters
    loss.backward()

    metric_logger.update(lr=optimizer.param_groups[0]['lr'])
    metric_logger.update(ce_loss=ce_loss.item())
    metric_logger.update(diff_loss=diff_loss.item())

    if log_writer is not None:
        """ We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        log_writer.log('ce_loss', ce_loss, commit=False)
        log_writer.log('diff_loss', diff_loss)


def loss_forward_and_backward_dsmil(net, image_patches, labels, criterion, conf,
                                    device, optimizer, metric_logger, log_writer=None):
    # Compute loss
    ins_preds, bag_preds, attn = net(image_patches)
    max_preds, _ = torch.max(ins_preds, 0, keepdim=True)
    ce_loss = 0.5 * criterion(max_preds, labels) \
              + 0.5 * criterion(bag_preds, labels)

    diff_loss = torch.tensor(0).to(device, dtype=torch.float)
    attn = torch.softmax(attn, dim=-1)
    for i in range(conf.n_token):
        for j in range(i + 1, conf.n_token):
            diff_loss += torch.cosine_similarity(attn[i], attn[j], dim=-1).mean() / (
                        conf.n_token * (conf.n_token - 1) / 2)

    loss = conf.w_loss * diff_loss + ce_loss

    # Backpropagate error and update parameters
    loss.backward()

    metric_logger.update(lr=optimizer.param_groups[0]['lr'])
    metric_logger.update(ce_loss=ce_loss.item())
    metric_logger.update(diff_loss=diff_loss.item())

    if log_writer is not None:
        """ We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        log_writer.log('ce_loss', ce_loss, commit=False)
        log_writer.log('diff_loss', diff_loss)


def loss_forward_and_backward_bmil(net, image_patches, coords, labels, criterion, conf,
                                   device, optimizer, metric_logger, log_writer=None):
    coords_array = coords.numpy()[0]
    # Compute loss
    logits, Y_prob, Y_hat, kl_div, _, _ = net(image_patches, coords_array, coords_array[:, 1].max(),
                                              coords_array[:, 0].max(), slide_label=labels)
    loss = criterion(logits, labels)
    kl_model = get_ard_reg_vdo(net)
    kl_div = kl_div.reshape(-1)
    kl_data = kl_div[0]
    loss += 1e-8 * kl_model + 1e-6 * kl_data
    # Backpropagate error and update parameters
    loss.backward()

    metric_logger.update(lr=optimizer.param_groups[0]['lr'])
    metric_logger.update(ce_loss=loss.item())

    if log_writer is not None:
        """ We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        log_writer.log('loss', loss)

def loss_forward_and_backward_clam(net, image_patches, labels, criterion, conf,
                                   device, optimizer, metric_logger, log_writer=None):
    # Compute loss
    logits, instance_loss = net(image_patches, labels, instance_eval=True)
    loss = criterion(logits, labels)
    total_loss = conf.w_loss * loss + (1 - conf.w_loss) * instance_loss


    # Backpropagate error and update parameters
    total_loss.backward()

    metric_logger.update(lr=optimizer.param_groups[0]['lr'])
    metric_logger.update(bag_loss=loss.item())
    metric_logger.update(instance_loss=instance_loss.item())

    if log_writer is not None:
        """ We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        log_writer.log('bag_loss', loss, commit=False)
        log_writer.log('instance_loss', instance_loss)


def train_one_epoch(net, criterion, data_loader, optimizer, device, epoch, conf, log_writer=None):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    net.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for data_it, data in enumerate(data_loader):
        # for data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        # Move input batch onto GPU if eager execution is enabled (default), else leave it on CPU
        # Data is a dict with keys `input` (patches) and `{task_name}` (labels for given task)
        image_patches = torch.load(data['input'][0], weights_only=True).to(device, dtype=torch.float32).unsqueeze(0)
        labels = data['label'].to(device)
        coords = data['coords']

        fft =  torch.tensor(np.load(data['fft'][0])['arr_0']).unsqueeze(0).to(device)

        # # Calculate and set new learning rate
        adjust_learning_rate(optimizer, epoch + data_it / len(data_loader), conf)
        optimizer.zero_grad()

        if conf.arch == 'dsmil':
            loss_forward_and_backward_dsmil(net, image_patches, labels, criterion, conf,
                                            device, optimizer, metric_logger, log_writer)
        elif conf.arch in ('clam_sb', 'clam_mb'):
            loss_forward_and_backward_clam(net, image_patches, labels, criterion, conf,
                                            device, optimizer, metric_logger, log_writer)
        elif conf.arch == 'bmil_spvis':
            loss_forward_and_backward_bmil(net, image_patches, coords, labels, criterion, conf,
                                           device, optimizer, metric_logger, log_writer)
        else:
            loss_forward_and_backward(net, image_patches, fft, labels, criterion, conf,
                                      device, optimizer, metric_logger, log_writer)

        optimizer.step()


# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(net, criterion, data_loader, device, conf, header, quick):
    # Set the network to evaluation mode
    net.eval()

    val_loss = 0
    test_labels = []
    test_predictions = []

    for data in data_loader:
        fft =  torch.tensor(np.load(data['fft'][0])['arr_0']).unsqueeze(0).to(device)
        image_patches = torch.load(data['input'][0], weights_only=True).to(device, dtype=torch.float32).unsqueeze(0)
        labels = data['label'].to(device)
        coords = data['coords']

        if conf.arch == 'dsmil':
            # Compute loss
            ins_preds, bag_preds, attn = net(image_patches)
            max_preds, _ = torch.max(ins_preds, 0, keepdim=True)
            loss = 0.5 * criterion(max_preds, labels) \
                   + 0.5 * criterion(bag_preds, labels)
            pred = 0.5 * torch.softmax(max_preds, dim=-1) \
                   + 0.5 * torch.softmax(bag_preds, dim=-1)
        elif conf.arch == 'bmil_spvis':
            coords_array = coords.numpy()[0]
            output, Y_prob, Y_hat, _, _ = net(image_patches, coords_array, coords_array[:, 1].max(),
                                              coords_array[:, 0].max(), validation=True)
            loss = criterion(output, labels)
            pred = torch.softmax(output, dim=-1)
        elif conf.arch in ('clam_sb', 'clam_mb'):
            output = net(image_patches)
            loss = criterion(output, labels)
            pred = torch.softmax(output, dim=-1)
        else:
            # Compute loss
            output = net(image_patches, fft)
            loss = criterion(output, labels)
            pred = torch.softmax(output, dim=-1)
        
        logits = output
        val_loss += loss.item()

        label = F.one_hot(labels.to('cpu'), num_classes=len(logits[0]))
        test_labels.extend([label.squeeze().cpu().numpy()])
        Y_prob = F.softmax(output, dim = 1)
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


