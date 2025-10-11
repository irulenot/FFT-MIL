import numpy as np
import torch
from utils.utils_fft import *
import os
import psutil
import time
import torch
import math
from dataset_modules.dataset_generic import save_splits
from models.model_mil_fft import mil_fft
from models.model_clam_fft import clam_fft
from models.model_clam_fft2 import clam_fft2
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import warnings
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.special import softmax
warnings.filterwarnings("ignore")
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train_noval(datasets, cur, args, device):
    """   
        train for a single fold
    """
    train_split, test_split = datasets
    print("Training on {} samples".format(len(train_split)))
    print("Testing on {} samples".format(len(test_split)))
    loss_fn = nn.CrossEntropyLoss()
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    if args.subtyping:
        model_dict.update({'subtyping': True})
    if args.B > 0:
        model_dict.update({'k_sample': args.B})
    model_dict.update({'n_classes': args.num_classes})
    instance_loss_fn = nn.CrossEntropyLoss()

    if args.model_type2 =='clam_fft':
        model = clam_fft(**model_dict, instance_loss_fn=instance_loss_fn)
    if args.model_type2 =='clam_fft2':
        model = clam_fft2(**model_dict, instance_loss_fn=instance_loss_fn)
    if args.model_type2 == "mil_fft":
        model = mil_fft(**model_dict)

    _ = model.to(device)

    print()

    optimizer = get_optim(model, args)
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    test_loader = get_split_loader(test_split, testing = args.testing)
    early_stopping = None

    save_dir = os.path.join('results/', args.split_dir.split('/')[-2], args.model_type2)
    print('save_dir:', save_dir)

    best_f1, best_roc_auc, best_loss, flag = 0, 0, math.inf, False
    for epoch in range(args.max_epochs):
        train_loop_noval(epoch, model, train_loader, optimizer, args.n_classes, device=device, loss_fn=loss_fn)
        test_loss_bag, metrics = validate_noval(cur, epoch, model, test_loader, args.n_classes, early_stopping = None, writer = None, loss_fn = loss_fn, results_dir=None, device=device)
        accuracy, precision_macro, recall_macro, f1_macro, report, conf_matrix, auc_roc, auc_roc_per_class = metrics
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

def test_noval(datasets, cur, args, device):
    """   
        train for a single fold
    """
    train_split, test_split = datasets
    print("Training on {} samples".format(len(train_split)))
    print("Testing on {} samples".format(len(test_split)))
    loss_fn = nn.CrossEntropyLoss()
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    if args.subtyping:
        model_dict.update({'subtyping': True})
    if args.B > 0:
        model_dict.update({'k_sample': args.B})
    model_dict.update({'n_classes': args.num_classes})
    instance_loss_fn = nn.CrossEntropyLoss()

    if args.model_type2 =='clam_fft':
        model = clam_fft(**model_dict, instance_loss_fn=instance_loss_fn)
    if args.model_type2 =='clam_fft2':
        model = clam_fft2(**model_dict, instance_loss_fn=instance_loss_fn)
    if args.model_type2 == "mil_fft":
        model = mil_fft(**model_dict)

    state_dict = torch.load(args.load_path)  # or "cuda" if using GPU
    model.load_state_dict(state_dict)
    _ = model.to(device)

    test_loader = get_split_loader(test_split, testing = args.testing)

    save_dir = os.path.join('results/', args.split_dir.split('/')[-2], args.model_type2)
    print('save_dir:', save_dir)

    test_labels_class, test_predictions_class, features = test_noval_loop(cur, 0, model, test_loader, args.n_classes, early_stopping = None, writer = None, loss_fn = loss_fn, results_dir=None, device=device)

    torch.save(features, "features/fft_feats.pt")
    torch.save(test_predictions_class, "features/fft_preds.pt")
    return None



def get_model_size(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params

def train_loop_noval(epoch, model, loader, optimizer, n_classes, device, writer = None, loss_fn = None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    # print('\n')
    for batch_idx, (data, fft, label) in enumerate(loader):
        label = torch.tensor([label])
        data, fft, label = data.to(device), fft.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data, fft)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

def validate_noval(cur, epoch, model, loader, n_classes, device=None, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    val_loss = 0
    test_labels = []
    test_predictions = []

    with torch.no_grad():
        for batch_idx, (data, fft, label) in enumerate(loader):
            label = torch.tensor([label])
            data, fft, label = data.to(device), fft.to(device), label.to(device)

            logits, Y_prob, Y_hat, _, _ = model(data, fft)
            
            label = F.one_hot(label, num_classes=n_classes).to(float)
            loss = loss_fn(logits, label)
            
            val_loss += loss.item()

            label = label.to('cpu')
            test_labels.extend([label.squeeze().cpu().numpy()])
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
    if len(classes) == 2:
        y_true_bin = np.eye(len(classes))[test_labels_class]
    else: 
        y_true_bin = label_binarize(test_labels_class, classes=classes)
        
    # Multi-class AUC-ROC using macro averaging
    auc_roc = roc_auc_score(y_true_bin, test_predictions, average='macro', multi_class='ovr')
    auc_roc_per_class = roc_auc_score(y_true_bin, test_predictions, average=None, multi_class='ovr')  # shape: (num_classes,)

    metrics = [accuracy, precision_macro, recall_macro, f1_macro, report, conf_matrix, auc_roc, auc_roc_per_class]

    return val_loss / len(loader), metrics

def test_noval_loop(cur, epoch, model, loader, n_classes, device=None, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    val_loss = 0
    test_labels = []
    test_predictions = []
    features = []

    with torch.no_grad():
        for batch_idx, (data, fft, label) in enumerate(loader):
            label = torch.tensor([label])
            data, fft, label = data.to(device), fft.to(device), label.to(device)

            logits, Y_prob, Y_hat, _, feature = model(data, fft, return_features=True)
            features.append(feature['features'].cpu().numpy())
            label = F.one_hot(label, num_classes=n_classes).to(float)

            test_labels.extend([label.squeeze().cpu().numpy()])
            test_predictions.extend([Y_prob.squeeze().cpu().numpy()])

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)    
    test_labels_class = np.argmax(test_labels, axis=1)
    test_predictions_class = np.argmax(test_predictions, axis=1)

    return test_labels_class, test_predictions_class, features

def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, fft, label) in enumerate(loader):
        label = torch.tensor([label])
        data, fft, label = data.to(device), fft.to(device), label.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data, fft)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger
