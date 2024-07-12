#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mwmak
Perform speech depression detection (SDD) on DAIC-WOZ using leave-one-speaker-out cross validation.
For illustration purpose, a simple 1D-CNN on MFCC or COVAREP features was used. Each speaker in the dataset 
is classified to normal ('nor') or depressed ('dep').

If --wtype is sess, one decision is made for each recording session. 
If --wtype is segs, one decision is made for each 3.84s-segment of a recording session.

"""

from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
from model import CNNModel
from pathlib import Path
import argparse
from dataset import FeatureDataset
import pandas as pd
import copy
from performance import get_f1_score, comp_loso_cv_f1, get_dec_count

def get_default_device():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device
        

import torch.nn.functional as F
def evaluate(model, loader, pos=1, neg=0):
    model.eval()
    batch = next(iter(loader))      # Only one batch for test
    [loss, tp, tn, fp, fn] = model.validation_step(batch, pos=pos, neg=neg)
    return {"loss" : loss.item(), "tp" : tp, "tn" : tn, "fp" : fp, "fn" : fn}


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(model, trn_loader, val_loader, n_epochs, lr, optimizer_function=torch.optim.Adam,
        class_weights=None):
    optimizer = optimizer_function(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=n_epochs, 
                                                steps_per_epoch=len(trn_loader))

    for epoch in range(n_epochs):
        # Train on training split of the CV
        model.train()
        lrs = []
        trn_loss = []
        trn_tp = trn_tn = trn_fp = trn_fn = 0
        tp = tn = fp = fn = 0
        for batch in tqdm(trn_loader, total=len(trn_loader)):
            loss = model.training_step(batch, class_weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            sched.step()
            l, tp, tn, fp, fn = model.validation_step(batch)
            trn_loss.append(l.cpu().numpy())
            trn_tp += tp ; trn_tn += tn ; trn_fp += fp ; trn_fn += fn

        trn_f1 = get_f1_score(trn_tp, trn_fp, trn_fn)
        trn_loss = np.mean(trn_loss)

        # Validate on the validation split of the CV. No need to compute F1 on validation
        # split as the test speaker belong to one class only
        l, _, _, _, _ = model.validation_step(next(iter(val_loader)))  # 1 batch in validation split
        print(f"Epoch: {epoch}, Trn loss: {trn_loss:0.4f}, Trn F1: {trn_f1:0.4f}, Val loss: {l:0.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool_method', choices=['sp', 'asp', 'avg'], required=True)
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--audiodir', default='../../audio')
    parser.add_argument('--trainlist', default='../../labels/ssd_labels_segs_train.txt')
    parser.add_argument('--testlist', default='../../labels/ssd_labels_segs_test.txt')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--wtype', choices=['segs', 'sess'], default='segs')
    parser.add_argument('--pos_class', choices=['dep', 'nor'], default='dep')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ftype', choices=['mfc', 'cov', 'mel'], default='mfc')

    args = parser.parse_args()
    n_classes = int(args.n_classes)
    max_len = 3.84*16000                  # Max utt duration in no. of samples (assume 16kHz)
    device = get_default_device()
    trainlist = args.trainlist
    testlist = args.testlist

    print(f'Using device {device}')
    print(f"No. of classes = {n_classes}")
    print(f'Feature type = {args.ftype}')

    # Get the training speaker ID list
    trn_df = pd.read_csv(trainlist, names=['seg_name', 'label', 'spk_id', 'gender'], sep=' ')
    trn_spk_list = trn_df['spk_id'].tolist()
    trn_spk_list = list(set(trn_spk_list))          # Make sure the speakers in the list are unique

    # Get the test speaker ID list
    tst_df = pd.read_csv(testlist, names=['seg_name', 'label', 'spk_id', 'gender'], sep=' ')
    tst_spk_list = tst_df['spk_id'].tolist()
    tst_spk_list = list(set(tst_spk_list))          # Make sure the speakers in the list are unique
    tst_set = FeatureDataset(filelist=testlist, audiodir=args.audiodir, n_classes=n_classes, 
                             max_len=max_len, spk_ids=tst_spk_list, device=device, ftype=args.ftype)

    pos = 1 if args.pos_class == 'dep' else 0
    neg = 1 if args.pos_class == 'nor' else 0

    # Perform leave-one-speaker-out cross-validation
    val_res = tst_res = []
    for spk in trn_spk_list:
        tmp_list = copy.deepcopy(trn_spk_list)  # Deep copy to avoid removing the current spk from trn_spk_list
        tmp_list.remove(int(spk))
        trn_spks = tmp_list         # Training speakers
        val_spk = [spk]             # Validation speaker

        trn_set = FeatureDataset(filelist=trainlist, audiodir=args.audiodir, n_classes=n_classes, 
                                 max_len=max_len, spk_ids=trn_spks, device=device, ftype=args.ftype)
        val_set = FeatureDataset(filelist=trainlist, audiodir=args.audiodir, n_classes=n_classes, 
                                 max_len=max_len, spk_ids=val_spk, device=device, ftype=args.ftype)

        n_trns = len(trn_set.df.index)
        n_vals = len(val_set.df.index)
        n_tsts = len(tst_set.df.index)
        trn_dl = DataLoader(trn_set, batch_size=args.batch_size, shuffle=True) 
        val_dl = DataLoader(val_set, batch_size=n_vals, shuffle=False)
        tst_dl = DataLoader(tst_set, batch_size=n_tsts, shuffle=False)
        
        # Get feature dimension
        input, label = next(iter(val_dl))
        fdim = input.shape[1]

        print(f"Validate on Speaker {spk}")
        print(f"No. of training samples = {n_trns}")
        print(f"No. of validation samples = {n_vals}")
        print(f"No. of test samples = {n_tsts}")
        print(f'Feature dimesion = {fdim}')
    
        # Initialize model
        model = CNNModel(pool_method=args.pool_method, n_inputs=fdim, 
                         n_classes=n_classes).to(device)

        # Train model and evaluate it on the validation set. The class weights are set to
        # address the class imbalance problem
        cweights = torch.tensor([0.3, 0.7]).to(device) 
        fit(model, trn_dl, val_dl, n_epochs=args.n_epochs, lr=0.0001, class_weights=cweights)

        # Accumulate the decisions on the validation split
        val_res.append(evaluate(model, val_dl, pos=pos, neg=neg))

        # Test model on the test set
        res = evaluate(model, tst_dl, pos=pos, neg=neg)
        tst_f1 = get_f1_score(res['tp'], res['fp'], res['fn'])
        print(f'F1 score on test set = {tst_f1:0.4f}')
        print(f"TP={res['tp']:0.4f}, TN={res['tn']:0.4f}, FP={res['fp']:0.4f}, FN={res['fn']:0.4f}")    
        tst_res.append(res)

        # Save the trained model to file
        Path("models").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.model_file)

    # Compute the average LOSOCV F1 on the validation splits for all speakers
    F1 = comp_loso_cv_f1(val_res)
    print(f'Average LOSOCV F1 on the validation splits of all speakers = {F1:.4f}')

    # Compute the average LOSOCV F1 on the test set
    F1 = comp_loso_cv_f1(tst_res)
    print(f'Average LOSOCV F1 on the test set = {F1:.4f}')



