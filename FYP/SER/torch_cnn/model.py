#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define network models
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from pooling import StatsPooling, AttenStatsPooling
from sklearn.metrics import accuracy_score, recall_score

class CNNModel(nn.Module):
    def __init__(self, n_inputs=40, n_classes=4, pool_method='None'):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_classes =  n_classes
        self.conv1 = nn.Sequential(nn.Conv1d(n_inputs, 16, kernel_size=15, padding=1),
                                   nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=5, padding=1),
                                   nn.ReLU(), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=5, padding=1),
                                   nn.ReLU(), nn.MaxPool1d(2))
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=5, padding=1),
                                   nn.ReLU(), nn.MaxPool1d(2))
        if pool_method == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((128,1))
            self.emb_size = 128            
        elif pool_method == 'sp':    
            self.pool = StatsPooling()
            self.emb_size = 256
        elif pool_method == 'asp':
            self.pool = AttenStatsPooling(in_dim=128, hid_dim=64, n_heads=1)
            self.emb_size = 256

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(self.emb_size, 64), nn.ReLU(), 
                                 nn.Linear(64, self.n_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.flatten(x)
        out = self.fc(x)
        return(out)

    def training_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        return loss
    
    def validation_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        _, pred = torch.max(outputs, 1)
        wa = accuracy_score(labels.cpu(), pred.cpu())   # Weighted accuracy
        ua = recall_score(labels.cpu(), pred.cpu(), average='macro') # Unweighted acc
        return [torch.mean(loss).detach(), wa, ua] 

