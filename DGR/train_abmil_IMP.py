import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support,f1_score
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.special import softmax

from models.classic_attmil import AttnMIL,GatedAttention
from models.dropout import LinearScheduler

from utils import *
from wsi_dataloader_3 import C16DatasetV3,C16DatasetV3_tcga,IMPDataset
from scipy import linalg
from timm.optim.adamp import AdamP
from radam import RAdam
from lookhead import Lookahead

def train(trainloader, milnet, criterion, optimizer, args):
    milnet.train()
    total_loss = 0
    bc = 0

    for batch_id, (feats, label) in enumerate(trainloader):
        bag_feats = feats.cuda()
        bag_label = label.cuda()
        bag_feats = bag_feats.view(-1, args.feats_size)

        optimizer.zero_grad()
        bag_prediction, _ = milnet(bag_feats)
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        loss = bag_loss 
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(milnet.parameters(),5.0)
        optimizer.step()
        total_loss = total_loss + loss.item()
        # sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (batch_id, len(trainloader), loss.item()))

    return total_loss / len(trainloader)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
def test(testloader, milnet, criterion, args):
    milnet.eval()
    # csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    
    with torch.no_grad():
        for batch_id, (feats, label) in enumerate(testloader):
            bag_feats = feats.cuda()
            bag_label = label.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)

            bag_prediction, _ = milnet(bag_feats)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            loss = bag_loss 
            total_loss = total_loss + loss.item()
            # sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (batch_id, len(testloader), loss.item()))
            test_labels.extend([label.squeeze().cpu().numpy()])
            Y_prob = F.softmax(bag_prediction, dim = 1)
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

    # Multi-class AUC-ROC using macro averaging
    auc_roc = roc_auc_score(y_true_bin, test_predictions, average='macro', multi_class='ovr')
    auc_roc_per_class = roc_auc_score(y_true_bin, test_predictions, average=None, multi_class='ovr')  # shape: (num_classes,)

    metrics = [accuracy, precision_macro, recall_macro, f1_macro, report, conf_matrix, auc_roc, auc_roc_per_class]
    return total_loss / len(testloader), metrics

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--dataroot', default="/data/fftmil/IMP/features_CLAM/pt_files/", type=str, help='dataroot for the CAMELYON16 dataset')
    parser.add_argument('--backgrd_thres', default=30, type=int, help='background threshold')
    parser.add_argument('--num_classes', default=3, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers used in dataloader [4]')
    parser.add_argument('--feats_size', default=1024, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, default=0, help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--seed', default='0', type=int, help='random seed')
    parser.add_argument('--drop_p', default=0.0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--model', default='attn', type=str, help='MIL model [dsmil]')
    parser.add_argument('--save_dir', default='results/abmil_imp/', type=str, help='the directory used to save all the output')
    parser.add_argument('--fold', default='0', type=str, help='fold')
    parser.add_argument('--quick', default=False, type=bool)
    args = parser.parse_args()
    #gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    torch.cuda.set_device(args.gpu_index)
    maybe_mkdir_p(join(args.save_dir, args.model))
    args.save_dir = make_dirs(join(args.save_dir, args.model))
    maybe_mkdir_p(args.save_dir)


    # <------------- set up logging ------------->
    logging_path = os.path.join(args.save_dir, 'Train_log.log')
    logger = get_logger(logging_path)

    # <------------- save hyperparams ------------->
    option = vars(args)
    file_name = os.path.join(args.save_dir, 'option.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    criterion = nn.BCEWithLogitsLoss()
    
    # <------------- define MIL network ------------->
    if args.model == "attn":
        milnet = AttnMIL(args.feats_size,args.num_classes,args.drop_p).cuda()

    elif args.model == 'gate':
        milnet = GatedAttention(args.feats_size,args.num_classes,args.drop_p).cuda()


    optimizer = torch.optim.AdamW(milnet.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    
    trainset = IMPDataset(args, 'train')
    testset = IMPDataset(args, 'test')

    trainloader = DataLoader(trainset, 1, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    testloader = DataLoader(testset, 1, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    import math
    best_f1, best_roc_auc, best_loss, flag = 0, 0, math.inf, False

    save_path = join(args.save_dir, 'weights')
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in tqdm(range(1, args.num_epochs + 1)):
        train_loss_bag = train(trainloader, milnet, criterion, optimizer, args) # iterate all bags
        test_loss_bag, metrics = test(testloader, milnet, criterion, args)

        accuracy, precision_macro, recall_macro, f1_macro, report, conf_matrix, auc_roc, auc_roc_per_class = metrics
        save_dir = './results/abmil_IMP/'
        os.makedirs(save_dir, exist_ok=True)
        if f1_macro > best_f1:
            print(f"F1: best loss: {test_loss_bag:.4f}, best f1: {f1_macro:.4f}, best acc: {accuracy:.4f}, best roc_auc: {auc_roc:.4f}")
            np.save(os.path.join(save_dir, 'metrics_f1.npy'), np.array(metrics, dtype=object))
            torch.save(milnet.state_dict(), os.path.join(save_dir, 'model_f1.pt'))
            best_f1 = f1_macro
            flag = True
        if auc_roc > best_roc_auc:
            print(f"AUC_ROC: best loss: {test_loss_bag:.4f}, best f1: {f1_macro:.4f}, best acc: {accuracy:.4f}, best roc_auc: {auc_roc:.4f}")
            np.save(os.path.join(save_dir, 'metrics_auc_roc.npy'), np.array(metrics, dtype=object))
            torch.save(milnet.state_dict(), os.path.join(save_dir, 'model_auc_roc.pt'))
            best_roc_auc = auc_roc
            flag = True
        if test_loss_bag < best_loss:
            print(f"LOSS: best loss: {test_loss_bag:.4f}, best f1: {f1_macro:.4f}, best acc: {accuracy:.4f}, best roc_auc: {auc_roc:.4f}")
            np.save(os.path.join(save_dir, 'metrics_loss.npy'), np.array(metrics, dtype=object))
            torch.save(milnet.state_dict(), os.path.join(save_dir, 'model_loss.pt'))
            best_loss = test_loss_bag
            flag = True
        if flag:
            print()
            flag = False

    print(f"Best F1: {best_f1:.4f}, Best AUC_ROC: {best_roc_auc:.4f}, Best Loss: {best_loss:.4f}")
    return None

if __name__ == '__main__':
    main()