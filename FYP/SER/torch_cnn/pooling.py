#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Define statistic pooling and attentive statistic pooling
'''

import torch.nn as nn
import torch.nn.functional as F
import torch

class StatsPooling(nn.Module):
    '''
    Statistics pooling
    # p. 21 of http://www.eie.polyu.edu.hk/~mwmak/papers/IEEE-Workshop-DL2023.pdf
    '''
    def __init__(self):
        super().__init__()

    def forward(self, h):
        mean = h.mean(-1, keepdim=True)
        var = torch.sqrt((h - mean).pow(2).mean(-1) + 1e-5)
        return torch.cat([mean.squeeze(-1), var], -1)

class AttenStatsPooling(nn.Module):
    '''
    Attentive statistics pooling
    # p. 23 of http://www.eie.polyu.edu.hk/~mwmak/papers/IEEE-Workshop-DL2023.pdf
    '''
    def __init__(self, in_dim, hid_dim, n_heads):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            torch.nn.ReLU(),
            nn.Linear(hid_dim, n_heads, bias=False)
        )
    def forward(self, h):
        scores = self.fc(h.transpose(-2, -1))
        alphas = F.softmax(scores, dim=-2).permute(-1, 0, 1)
        stats = []
        for alpha in alphas:
            stats.append(self.get_mean_var(h, alpha))
        return torch.cat(stats, -1)

    @staticmethod
    def get_mean_var(h, alpha):
        alpha = alpha[:, None, :]
        mu = h * alpha
        mu = mu.sum(-1,  keepdim=True)
        sigma = (h - mu).pow(2) * alpha
        sigma = torch.sqrt(sigma.sum(-1) + 1e-5)
        return torch.cat([mu.squeeze(-1), sigma], -1)     
    
if __name__ == "__main__":
    sp = StatsPooling()
    asp = AttenStatsPooling(5, 2, 1)
    input = torch.rand(2, 5, 9)        # Input: 2 batches, each with nine 5-dim vectors
    output = sp.forward(input)
    print(output)
    print(output.size())                # Output: 2 10-dim vectors

    output = asp.forward(input)
    print(output)
    print(output.size())                # Output: 1 vector with 20 dimensions


