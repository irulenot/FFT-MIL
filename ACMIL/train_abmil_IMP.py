
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
from datasets.datasets import build_HDF5_feat_dataset
from architecture.transformer import MHA, ABMIL
from architecture.transMIL import TransMIL
from engine import train_one_epoch, evaluate
from architecture.dsmil import MILNet, FCLayer, BClassifier
from architecture.bmil import probabilistic_MIL_Bayes_spvis
from architecture.clam import CLAM_SB, CLAM_MB
from architecture.ilra import ILRA
from modules import mean_max
import wandb

import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.special import softmax
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def get_arguments():
    parser = argparse.ArgumentParser('Patch classification training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/imp_config.yml',
                        help='settings of dataset in yaml format')
    parser.add_argument(
        "--seed", type=int, default=5, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--wandb_mode', default='disabled', choices=['offline', 'online', 'disabled'],
                        help='the model of wandb')
    parser.add_argument(
        "--w_loss", type=float, default=1.0, help="number of query token"
    )
    parser.add_argument(
        "--arch", type=str, default='abmil', choices=['transmil', 'clam_sb', 'clam_mb', 'abmil', 'ilra',
                                                 'mha', 'dsmil', 'bmil_spvis', 'meanmil', 'maxmil', 'acmil'], help="number of query token"
    )
    parser.add_argument('--pretrain', default='medical_ssl',
                        choices=['natural_supervsied', 'medical_ssl', 'plip', 'path-clip-B-AAAI'
                                                                              'path-clip-B', 'path-clip-L-336',
                                 'openai-clip-B', 'openai-clip-L-336', 'quilt-net', 'biomedclip', 'path-clip-L-768',
                                 'UNI', 'GigaPath'],
                        help='settings of Tip-Adapter in yaml format')
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate"
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

    # Prepare dataset
    set_seed(args.seed)

    # define datasets and dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset(os.path.join(conf.data_dir, 'patch_feats_pretrain_%s.h5'%conf.pretrain), conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    # define network
    if conf.arch == 'transmil':
        net = TransMIL(conf)
    elif conf.arch == 'mha':
        net = MHA(conf)
    elif conf.arch == 'clam_sb':
        net = CLAM_SB(conf).to(device)
    elif conf.arch == 'clam_mb':
        net = CLAM_MB(conf).to(device)
    elif conf.arch == 'dsmil':
        i_classifier = FCLayer(conf.D_feat, conf.n_class)
        b_classifier = BClassifier(conf, nonlinear=False)
        net = MILNet(i_classifier, b_classifier)
    elif conf.arch == 'bmil_spvis':
        net = probabilistic_MIL_Bayes_spvis(conf)
        net.relocate()
    elif conf.arch == 'abmil':
        net = ABMIL(conf)
    elif conf.arch == 'meanmil':
        net = mean_max.MeanMIL(conf).to(device)
    elif conf.arch == 'maxmil':
        net = mean_max.MaxMIL(conf).to(device)
    elif conf.arch == 'ilra':
        net = ILRA(feat_dim=conf.D_feat, n_classes=conf.n_class, ln=True)
    else:
        print("architecture %s is not exist."%conf.arch)
        sys.exit(1)
    net.to(device)

    criterion = nn.CrossEntropyLoss()

    # define optimizer, lr not important at this point
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=conf.lr, weight_decay=conf.wd)

    import math
    best_f1, best_roc_auc, best_loss, flag = 0, 0, math.inf, False

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    for epoch in range(conf.train_epoch):

        train_one_epoch(net, criterion, train_loader, optimizer, device, epoch, conf)

        # val_auc, val_acc, val_f1, val_loss = evaluate(net, criterion, val_loader, device, conf, 'Val')
        test_loss_bag, metrics = evaluate(net, criterion, test_loader, device, conf, 'Test')
        accuracy, precision_macro, recall_macro, f1_macro, report, conf_matrix, auc_roc, auc_roc_per_class = metrics
        save_dir = './results/abmil_IMP/'
        os.makedirs(save_dir, exist_ok=True)
        if f1_macro > best_f1:
            print(f"F1: best loss: {test_loss_bag:.4f}, best f1: {f1_macro:.4f}, best acc: {accuracy:.4f}, best roc_auc: {auc_roc:.4f}")
            np.save(os.path.join(save_dir, 'metrics_f1.npy'), np.array(metrics, dtype=object))
            torch.save(net.state_dict(), os.path.join(save_dir, 'model_f1.pt'))
            best_f1 = f1_macro
            flag = True
        if auc_roc > best_roc_auc:
            print(f"AUC_ROC: best loss: {test_loss_bag:.4f}, best f1: {f1_macro:.4f}, best acc: {accuracy:.4f}, best roc_auc: {auc_roc:.4f}")
            np.save(os.path.join(save_dir, 'metrics_auc_roc.npy'), np.array(metrics, dtype=object))
            torch.save(net.state_dict(), os.path.join(save_dir, 'model_auc_roc.pt'))
            best_roc_auc = auc_roc
            flag = True
        if test_loss_bag < best_loss:
            print(f"LOSS: best loss: {test_loss_bag:.4f}, best f1: {f1_macro:.4f}, best acc: {accuracy:.4f}, best roc_auc: {auc_roc:.4f}")
            np.save(os.path.join(save_dir, 'metrics_loss.npy'), np.array(metrics, dtype=object))
            torch.save(net.state_dict(), os.path.join(save_dir, 'model_loss.pt'))
            best_loss = test_loss_bag
            flag = True
        if flag:
            print()
            flag = False

    print(f"Best F1: {best_f1:.4f}, Best AUC_ROC: {best_roc_auc:.4f}, Best Loss: {best_loss:.4f}")
    return None


if __name__ == '__main__':
    main()
