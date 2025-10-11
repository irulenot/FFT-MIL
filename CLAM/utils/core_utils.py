import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import warnings
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.special import softmax
warnings.filterwarnings("ignore")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import psutil
import time
from ptflops import get_model_complexity_info

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

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    # writer_dir = os.path.join(args.results_dir, str(cur))
    # if not os.path.isdir(writer_dir):
    #     os.mkdir(writer_dir)

    # if args.log_data:
    #     from tensorboardX import SummaryWriter
    #     writer = SummaryWriter(writer_dir, flush_secs=15)

    # else:
    #     writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        model_dict.update({'n_classes': args.num_classes})

        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    best_auc = 0
    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop, best_auc = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, loss_fn, args.results_dir, best_auc=best_auc)
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, loss_fn)
            stop, best_auc = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, loss_fn, args.results_dir, best_auc=best_auc)
        
        if stop: 
            break

    # if args.early_stopping:
    #     model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    # else:
    #     torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 

def train_noval(datasets, cur, args):
    """   
        train for a single fold
    """
    # print('\nTraining Fold {}!'.format(cur))
    # writer_dir = os.path.join(args.results_dir, str(cur))
    # if not os.path.isdir(writer_dir):
    #     os.mkdir(writer_dir)
    writer = None
    # print('\nInit train/test splits...', end=' ')
    train_split, test_split = datasets
    # save_splits(datasets, ['train', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    # print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Testing on {} samples".format(len(test_split)))
    # print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    # print('Done!')
    # print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        model_dict.update({'n_classes': args.num_classes})
        instance_loss_fn = nn.CrossEntropyLoss()
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        # elif args.model_type == 'fft':
        #     model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)
    # print('Done!')
    # print_network(model)
    # print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    # print('Done!')
    # print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    test_loader = get_split_loader(test_split, testing = args.testing)
    # print('Done!')
    # print('\nSetup EarlyStopping...', end=' ')
    early_stopping = None
    # print('Done!')

    save_dir = os.path.join('results/', args.split_dir.split('/')[-2], args.model_type)
    print('save_dir:', save_dir)

    best_f1, best_roc_auc, best_loss, flag = 0, 0, math.inf, False
    for epoch in range(args.max_epochs):
        train_loop_noval(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        test_loss_bag, metrics = validate_noval(cur, epoch, model, test_loader, args.n_classes, early_stopping = None, writer = None, loss_fn = loss_fn, results_dir=None)
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

def train_noval_timed(datasets, cur, args):
    train_split, test_split = datasets
    print("Training on {} samples".format(len(train_split)))
    print("Testing on {} samples".format(len(test_split)))

    loss_fn = nn.CrossEntropyLoss()

    model_dict = {"dropout": args.drop_out,
                  'n_classes': args.n_classes,
                  "embed_dim": args.embed_dim}
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        model_dict.update({'n_classes': args.num_classes})
        instance_loss_fn = nn.CrossEntropyLoss()
        if args.model_type == 'clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    else:
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = get_optim(model, args)
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    test_loader = get_split_loader(test_split, testing=args.testing)

    save_dir = os.path.join('results/', args.split_dir.split('/')[-2], args.model_type)
    print('save_dir:', save_dir)

    # -- Report Model Size --
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # -- Resource tracking --
    process = psutil.Process(os.getpid())
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    total_epoch_time = 0.0
    total_inference_time = 0.0
    total_train_samples = 0
    total_train_time = 0.0

    best_f1, best_roc_auc, best_loss = 0, 0, math.inf

    for epoch in range(args.max_epochs):
        torch.cuda.reset_peak_memory_stats()
        epoch_start = time.time()

        train_sample_count = len(train_loader.dataset)
        total_train_samples += train_sample_count

        train_loop_noval(epoch, model, train_loader, optimizer, args.n_classes, writer=None, loss_fn=loss_fn)
        epoch_train_time = time.time() - epoch_start
        total_train_time += epoch_train_time

        inf_start = time.time()
        test_loss_bag, metrics = validate_noval(
            cur, epoch, model, test_loader, args.n_classes,
            early_stopping=None, writer=None, loss_fn=loss_fn, results_dir=None
        )
        inference_time = time.time() - inf_start
        total_inference_time += inference_time

        epoch_time = time.time() - epoch_start
        total_epoch_time += epoch_time
        peak_gpu_epoch = torch.cuda.max_memory_allocated() / (1024 ** 2)

        accuracy, precision_macro, recall_macro, f1_macro, report, conf_matrix, auc_roc, auc_roc_per_class = metrics

        print(f"[Epoch {epoch:02d}] Time: {epoch_time:.2f}s | Inference: {inference_time:.2f}s | Peak GPU: {peak_gpu_epoch:.2f} MB")

        if f1_macro > best_f1:
            best_f1 = f1_macro
        if auc_roc > best_roc_auc:
            best_roc_auc = auc_roc
        if test_loss_bag < best_loss:
            best_loss = test_loss_bag

        # Final reporting
        total_runtime = time.time() - start_time
        avg_epoch_time = total_epoch_time / args.max_epochs
        avg_inference_time = total_inference_time / args.max_epochs
        inference_per_sample = avg_inference_time / len(test_loader.dataset)
        cpu_mem_MB = process.memory_info().rss / (1024 ** 2)
        peak_gpu_total_MB = torch.cuda.max_memory_allocated() / (1024 ** 2)
        training_throughput = total_train_samples / total_train_time

        print(f"\n[Final Resource Usage]")
        print(f"Total Runtime         : {total_runtime:.2f} s")
        print(f"Average Epoch Time    : {avg_epoch_time:.2f} s")
        print(f"Average Inference Time: {avg_inference_time:.2f} s")
        print(f"Inference per sample  : {inference_per_sample:.4f} s")
        print(f"CPU Memory Usage      : {cpu_mem_MB:.2f} MB")
        print(f"Peak GPU RAM          : {peak_gpu_total_MB:.2f} MB")
        print(f"Training Throughput   : {training_throughput:.2f} samples/sec")
        print(f"Model Parameters      : {num_params / 1e6:.2f} million")

        print(f"\n[Best Metrics]")
        print(f"Best F1               : {best_f1:.4f}")
        print(f"Best AUC_ROC          : {best_roc_auc:.4f}")
        print(f"Best Loss             : {best_loss:.4f}")

def test_noval(datasets, cur, args):
    """   
        train for a single fold
    """
    # print('\nTraining Fold {}!'.format(cur))
    # writer_dir = os.path.join(args.results_dir, str(cur))
    # if not os.path.isdir(writer_dir):
    #     os.mkdir(writer_dir)
    writer = None
    # print('\nInit train/test splits...', end=' ')
    train_split, test_split = datasets
    # save_splits(datasets, ['train', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    # print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Testing on {} samples".format(len(test_split)))
    # print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    # print('Done!')
    # print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        model_dict.update({'n_classes': args.num_classes})
        instance_loss_fn = nn.CrossEntropyLoss()
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        # elif args.model_type == 'fft':
        #     model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    state_dict = torch.load(args.load_path)  # or "cuda" if using GPU
    model.load_state_dict(state_dict)
    _ = model.to(device)
    # print('Done!')
    # print_network(model)
    # print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    # print('Done!')
    # print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    test_loader = get_split_loader(test_split, testing = args.testing)
    # print('Done!')
    # print('\nSetup EarlyStopping...', end=' ')
    early_stopping = None
    # print('Done!')

    test_labels_class, test_predictions_class, features = test_noval_loop(cur, 0, model, test_loader, args.n_classes, early_stopping = None, writer = None, loss_fn = loss_fn, results_dir=None)

    torch.save(features, "features/feats.pt")
    torch.save(test_predictions_class, "features/preds.pt")
    torch.save(test_labels_class, "features/labels.pt")
    return None

def train_loop_clam_noval(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    model.train()
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0
    # print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 
        train_loss += loss_value
        error = calculate_error(Y_hat, label)
        train_error += error
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        # if (batch_idx + 1) % 20 == 0:
        #     print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
        #         'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            # print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        # print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
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

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

def train_loop_noval(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    # print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        # if (batch_idx + 1) % 20 == 0:
        #     print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
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

    # print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    # for i in range(n_classes):
    #     acc, correct, count = acc_logger.get_summary(i)
    #     print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
    #     if writer:
    #         writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    # if writer:
    #     writer.add_scalar('train/loss', train_loss, epoch)
    #     writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None, best_auc=0):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    if auc > best_auc:
        best_auc = auc
        print('best_auc', best_auc)
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    # if early_stopping:
    #     assert results_dir
    #     early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
    #     if early_stopping.early_stop:
    #         print("Early stopping")
    #         return True

    return False, best_auc

def validate_noval(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    val_loss = 0
    test_labels = []
    test_predictions = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)
            
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
    auc_roc = roc_auc_score(y_true_bin, test_predictions, average='macro', multi_class='ovo')
    auc_roc_per_class = roc_auc_score(y_true_bin, test_predictions, average=None, multi_class='ovo')  # shape: (num_classes,)

    metrics = [accuracy, precision_macro, recall_macro, f1_macro, report, conf_matrix, auc_roc, auc_roc_per_class]

    return val_loss / len(loader), metrics

def test_noval_loop(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    val_loss = 0
    test_labels = []
    test_predictions = []
    features = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, feature = model(data, return_features=True)
            features.append(feature['features'].cpu().numpy())
            label = F.one_hot(label, num_classes=n_classes).to(float)

            test_labels.extend([label.squeeze().cpu().numpy()])
            test_predictions.extend([Y_prob.squeeze().cpu().numpy()])

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)    
    test_labels_class = np.argmax(test_labels, axis=1)
    test_predictions_class = np.argmax(test_predictions, axis=1)

    return test_labels_class, test_predictions_class, features

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None, best_auc=0):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)
    if auc > best_auc:
        best_auc = auc
    print('best_auc', best_auc)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    # if early_stopping:
    #     assert results_dir
    #     early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
    #     if early_stopping.early_stop:
    #         print("Early stopping")
    #         return True

    return False, best_auc

def validate_clam_noval(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None, best_auc=0):
    model.eval()
    val_loss = 0
    test_labels = []
    test_predictions = []
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            
            loss = loss_fn(logits, label)
            val_loss += loss.item()

            label = F.one_hot(label.to('cpu'), num_classes=n_classes)
            test_labels.extend([label.squeeze().cpu().numpy()])
            test_predictions.extend([torch.sigmoid(logits).squeeze().cpu().numpy()])

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)    
    test_labels_class = np.argmax(test_labels, axis=1)
    test_predictions_class = np.argmax(test_predictions, axis=1)
    f1 = f1_score(test_labels_class, test_predictions_class, average='weighted')
    acc = accuracy_score(test_labels_class, test_predictions_class)

    test_predictions_pred = softmax(test_predictions)
    roc_auc = roc_auc_score(test_labels, test_predictions_pred, multi_class='ovr')

    return val_loss / len(loader), acc, f1, roc_auc

def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

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
