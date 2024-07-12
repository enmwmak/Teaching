'''
Define Dataset classes
'''
from torch.utils.data import Dataset
import pandas as pd
import librosa
import numpy as np
import torch

class FeatureDataset(Dataset):
    '''
    Read the MFCC or COVAREP (.npy) files directly, which is much faster
    and the GPU utilization will near 100%. Need to run wav2mfc.py or extract_cov.py first
    '''
    def __init__(self, filelist=None, audiodir=None, max_len=61440,
                 n_classes=4, spk_ids=np.arange(1,10), device='cpu', ftype='mfc'):
        sdd_set = {'2' : ['nor','dep']}
        self.filelist = filelist
        self.spk_ids = spk_ids
        self.device = device
        df = pd.read_csv(filelist, names=['seg_name', 'label', 'spk_id', 'gender'], sep=' ')
        df = df[(df['spk_id'].isin(spk_ids))]
        self.audiodir = audiodir
        self.max_len = max_len                   # Max length in samples
        self.max_frm = int(max_len/160)          # hop length = 160 (100Hz frame rate)
        self.n_classes = n_classes
        self.df = df[(df['label'].isin(sdd_set[str(n_classes)]))]
        self.labels = dict()
        self.ftype = ftype

        for i, l in enumerate(sdd_set[str(n_classes)]):
            self.labels[l] = i

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        label = self.labels[row['label']]
        feafile = self.audiodir + '/' + self.ftype + '/' + row['seg_name'] + '.npy'
        feat = np.load(feafile)
        self.fdim, cur_frm = feat.shape

        # Data loader only supports items of the same size unless we provide our own collate_fn()
        if cur_frm >= self.max_frm:
            out_feat = feat[:, 0:self.max_frm]      # Trim the tail
        else:
            out_feat = np.zeros((self.fdim, self.max_frm))     # Pad tail with 0     
            n_repeats = int(self.max_frm / cur_frm) # Repeat the waveform N times to fill the array
            for k in range(0, n_repeats):
                out_feat[:, k*cur_frm : (k+1)*cur_frm] = feat

        feat = torch.from_numpy(np.array(out_feat, dtype=np.float32)).to(self.device)
        label = np.array(label, dtype=np.float32)
        label = torch.tensor(np.array(label, dtype=np.float32)).to(self.device)
        return feat, label
    

