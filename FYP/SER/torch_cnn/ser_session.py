#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mwmak
Perform speech emotion recognition (SER) on IEMOCAP using leave-one-session-out cross validation.
In each turn, four sessions (with eight speakers) are used as training data and one session (with
two speakers) is used as test data. For illustration purpose, a simple 1D-CNN on 
MFCC features was used.

Example usage:
    python3 ser_session.py --pool_method sp --model_file models/emotion_cnn_stats.pth
    python3 ser_session.py --pool_method avg --model_file models/emotion_cnn_avg.pth
    python3 ser_session.py --pool_method asp --model_file models/emotion_cnn_attend.pth

If using WaveMfcDataset in dataset.py, you need to run wav2mfc.py to conver the .wav files
to .npy files first
    
Performance:
    --pool_method sp --n_mfcc 20 --n_epochs 10
    LOSOCV Weighted Accuracy (WA) = 42.21%
    LOSOCV Unweighted Accuracy (UA) = 37.76%
"""

from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
from model import CNNModel
from pathlib import Path
import argparse
from performance import get_accuracy
from dataset import SessionMfcDataset

def get_default_device():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device
        

def evaluate(model, loader):
    model.eval()
    batch = next(iter(loader))      # Only one batch for test
    [loss, wa, ua] = model.validation_step(batch)
    return {"loss" : loss.item(), "wa" : wa, "ua" : ua}


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(model, train_loader, val_loader, n_epochs, lr, optimizer_function=torch.optim.Adam):
    history = []
    optimizer = optimizer_function(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=n_epochs, 
                                                steps_per_epoch=len(train_loader))

    for epoch in range(n_epochs):
        print("Epoch ", epoch)

        #Train
        model.train()
        lrs = []
        tr_loss = []
        for batch in tqdm(train_loader, total=len(train_loader)):
            loss = model.training_step(batch)
            tr_loss.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            sched.step()

        #Validate
        result = evaluate(model, val_loader)
        result["lrs"] = lrs
        result["train loss"] = torch.stack(tr_loss).mean().item()
        wa = result['wa']*100
        ua = result['ua']*100
        print(f"Last lr: {lrs[-1]:.2e}, Train_loss: {result['train loss']:.2f}, Val_loss: {result['loss']:.2f}, WA: {wa:.2f}%, UA: {ua:.2f}%")
        history.append(result)                 
    return history

def get_loso_cv_acc(results):
    wa = ua = 0
    for result in results:
        wa += result["wa"]
        ua += result["ua"]
    wa = wa/len(results)
    ua = ua/len(results)
    return wa, ua

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool_method', choices=['sp', 'asp', 'avg'], default='asp')
    parser.add_argument('--model_file', default='models/emotion_cnn_asp.pth')
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument('--rootdir', default='../../IEMOCAP_full_release')
    parser.add_argument('--filelist', default='../../labels/emo_labels_cat_exc-as-hap.txt')
    parser.add_argument('--n_mfcc', type=int, default=20)
    parser.add_argument('--n_epochs', type=int, default=10)

    args = parser.parse_args()
    n_classes = int(args.n_classes)
    max_len = 10*16000                  # Max utt duration in no. of samples (assume 16kHz)
    device = get_default_device()
    print(f'Using device {device}')

    # Perform leave-one-session-out cross-validation
    results = []
    sess_list = np.arange(1,6)      # IEMOCAP has 5 sessions
    for k in range(1,6):
        trn_sess = np.delete(sess_list, k-1)
        tst_sess = np.arange(k, k+1)

        # For each fold, use 4 sessions for training and the remaining session for test
        train_set = SessionMfcDataset(filelist=args.filelist, rootdir=args.rootdir, n_mfcc=args.n_mfcc,
                                    n_classes=n_classes, max_len=max_len, sess_ids=trn_sess,
                                    device=device)
        test_set = SessionMfcDataset(filelist=args.filelist, rootdir=args.rootdir, n_mfcc=args.n_mfcc,
                                    n_classes=n_classes, max_len=max_len, sess_ids=tst_sess,
                                    device=device)

        n_trains = len(train_set.df.index)
        n_tests = len(test_set.df.index)
        train_dl = DataLoader(train_set, batch_size=64, shuffle=True) 
        test_dl = DataLoader(test_set, batch_size=n_tests, shuffle=False)

        print(f"Fold {k}")
        print(f"No. of classes = {n_classes}")
        print(f"No. of training samples = {n_trains}")
        print(f"No. of test samples = {n_tests}")
        print(f"Training sessions: {train_set.sess_ids}");
        print(f"Test session: {test_set.sess_ids}");
    
        # Initialize model
        model = CNNModel(pool_method=args.pool_method, n_inputs=args.n_mfcc*3, 
                         n_classes=n_classes).to(device)

        # Train model
        fit(model, train_dl, test_dl, n_epochs=args.n_epochs, lr=0.0001)

        # Test model
        result = evaluate(model, test_dl)
        print(f'Weighted Accuracy (WA) on Session {k} = {result["wa"]*100:.2f}%')
        print(f'Unweighted Accuracy (UA) on Session {k} = {result["ua"]*100:.2f}%')
        results.append(result)

        # Save the trained model to file
        Path("models").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.model_file)

    # Compute the LOSOCV WA and UA
    WA, UA = get_loso_cv_acc(results)
    print(f'LOSOCV Weighted Accuracy (WA) = {WA*100:.2f}%')
    print(f'LOSOCV Unweighted Accuracy (UA) = {UA*100:.2f}%')



