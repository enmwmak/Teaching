'''
Define Dataset classes
'''
from torch.utils.data import Dataset
import pandas as pd
import librosa
import numpy as np
import torch

class SpeakerWavDataset(Dataset):
    '''
    Use Librosa to convert waveform to MFCC, which is slow because each .wav file needs 
    to be converted multiple times during training of the CNN.
    Dataset class for leave-one-speaker-out CV
    '''
    def __init__(self, filelist=None, rootdir=None, n_mfcc=20, max_len=64000,
                 n_classes=4, spk_ids=np.arange(1,10), device='cpu'):
        emo_set = {'2' : ['ang','neu'],
                   '4' : ['ang','hap','sad','neu'],
                   '5' : ['ang','hap','sad','neu', 'exc']}
        self.filelist = filelist
        self.spk_ids = spk_ids
        self.device = device
        df = pd.read_csv(filelist, names=['file', 'label', 'spk_id', 'gender'], sep=' ')
        df = df[(df['spk_id'].isin(spk_ids))]
        self.rootdir = rootdir
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.n_classes = n_classes
        self.df = df[(df['label'].isin(emo_set[str(n_classes)]))]
        self.labels = dict()
        for i, emo in enumerate(emo_set[str(n_classes)]):
            self.labels[emo] = i

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        label = self.labels[row['label']]
        wavefile = self.rootdir + '/' + row['file'] + '.wav'
        waveform, srate = librosa.load(wavefile)
        cur_len = waveform.shape[0]

        # Data loader only supports items of the same size unless we provide our own collate_fn()
        if cur_len > self.max_len:
            out_wav = waveform[0:self.max_len]      # Trim the tail
        else:
            out_wav = np.zeros((self.max_len,))     # Pad tail with 0     
            n_repeats = int(self.max_len / cur_len) # Repeat the waveform N times to fill the array
            for k in range(0, n_repeats):
                out_wav[k*cur_len : (k+1)*cur_len] = waveform

        mfcc = librosa.feature.mfcc(y=out_wav, sr=srate, n_mfcc=self.n_mfcc,
                                    n_fft=512, hop_length=160, center=False)
        dmfcc = librosa.feature.delta(mfcc)
        ddmfcc = librosa.feature.delta(dmfcc)
        mfcc = np.vstack((mfcc, dmfcc, ddmfcc))
        mfcc = torch.from_numpy(np.array(mfcc, dtype=np.float32)).to(self.device)
        label = np.array(label, dtype=np.float32)
        label = torch.tensor(np.array(label, dtype=np.float32)).to(self.device)
        return mfcc, label


class SessionWavDataset(Dataset):
    '''
    Use Librosa to convert waveform to MFCC, which is slow because each .wav file needs 
    to be converted multiple times during training of the CNN.
    Dataset class for leave-one-session-out CV
    '''
    def __init__(self, filelist=None, rootdir=None, n_mfcc=20, max_len=64000,
                 n_classes=4, sess_ids=np.arange(1,5), device='cpu'):
        emo_set = {'2' : ['ang','neu'],
                   '4' : ['ang','hap','sad','neu'],
                   '5' : ['ang','hap','sad','neu', 'exc']}
        self.filelist = filelist
        self.sess_ids = sess_ids
        self.device = device
        df = pd.read_csv(filelist, names=['file', 'label', 'spk_id', 'gender'], sep=' ')
        files = df['file'].tolist()
        sessions = []
        for file in files:
            sessions.append(int(file.split("/", 1)[0][-1]))
        df = df.assign(sess_id=sessions)    
        df = df[(df['sess_id'].isin(sess_ids))]
        self.rootdir = rootdir
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.n_classes = n_classes
        self.df = df[(df['label'].isin(emo_set[str(n_classes)]))]
        self.labels = dict()
        for i, emo in enumerate(emo_set[str(n_classes)]):
            self.labels[emo] = i

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        label = self.labels[row['label']]
        wavefile = self.rootdir + '/' + row['file'] + '.wav'
        waveform, srate = librosa.load(wavefile)
        cur_len = waveform.shape[0]

        # Data loader only supports items of the same size unless we provide our own collate_fn()
        if cur_len > self.max_len:
            out_wav = waveform[0:self.max_len]      # Trim the tail
        else:
            out_wav = np.zeros((self.max_len,))     # Pad tail with 0     
            n_repeats = int(self.max_len / cur_len) # Repeat the waveform N times to fill the array
            for k in range(0, n_repeats):
                out_wav[k*cur_len : (k+1)*cur_len] = waveform

        mfcc = librosa.feature.mfcc(y=out_wav, sr=srate, n_mfcc=self.n_mfcc,
                                    n_fft=512, hop_length=160, center=False)
        dmfcc = librosa.feature.delta(mfcc)
        ddmfcc = librosa.feature.delta(dmfcc)
        mfcc = np.vstack((mfcc, dmfcc, ddmfcc))
        mfcc = torch.from_numpy(np.array(mfcc, dtype=np.float32)).to(self.device)
        label = np.array(label, dtype=np.float32)
        label = torch.tensor(np.array(label, dtype=np.float32)).to(self.device)
        return mfcc, label


class SpeakerMfcDataset(Dataset):
    '''
    Read the MFCC (.npy) files directly without converting from waveform, which is much faster
    and the GPU utilization will near 100%. Need to run wav2mfc.py first
    Dataset class for leave-one-speaker-out CV
    '''
    def __init__(self, filelist=None, rootdir=None, n_mfcc=20, max_len=64000,
                 n_classes=4, spk_ids=np.arange(1,10), device='cpu'):
        emo_set = {'2' : ['ang','neu'],
                   '4' : ['ang','hap','sad','neu'],
                   '5' : ['ang','hap','sad','neu', 'exc']}
        self.filelist = filelist
        self.spk_ids = spk_ids
        self.device = device
        df = pd.read_csv(filelist, names=['file', 'label', 'spk_id', 'gender'], sep=' ')
        df = df[(df['spk_id'].isin(spk_ids))]
        self.rootdir = rootdir
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.max_frm = int(max_len/160)          # hop length = 160
        self.n_classes = n_classes
        self.df = df[(df['label'].isin(emo_set[str(n_classes)]))]
        self.labels = dict()
        for i, emo in enumerate(emo_set[str(n_classes)]):
            self.labels[emo] = i

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        label = self.labels[row['label']]
        mfccfile = self.rootdir + '/' + row['file'].replace('/wav/', '/mfc/') + '.npy'
        mfcc = np.load(mfccfile)
        self.n_mfcc, cur_frm = mfcc.shape

        # Data loader only supports items of the same size unless we provide our own collate_fn()
        if cur_frm > self.max_frm:
            out_mfcc = mfcc[:, 0:self.max_frm]      # Trim the tail
        else:
            out_mfcc = np.zeros((self.n_mfcc, self.max_frm))     # Pad tail with 0     
            n_repeats = int(self.max_frm / cur_frm) # Repeat the waveform N times to fill the array
            for k in range(0, n_repeats):
                out_mfcc[:, k*cur_frm : (k+1)*cur_frm] = mfcc

        mfcc = torch.from_numpy(np.array(out_mfcc, dtype=np.float32)).to(self.device)
        label = np.array(label, dtype=np.float32)
        label = torch.tensor(np.array(label, dtype=np.float32)).to(self.device)
        return mfcc, label
    

class SessionMfcDataset(Dataset):
    '''
    Read the MFCC (.npy) files directly without converting from waveform, which is much faster
    and the GPU utilization will near 100%. Need to run wav2mfc.py first
    Dataset class for leave-one-session-out CV
    '''
    def __init__(self, filelist=None, rootdir=None, n_mfcc=20, max_len=64000,
                 n_classes=4, sess_ids=np.arange(1,5), device='cpu'):
        emo_set = {'2' : ['ang','neu'],
                   '4' : ['ang','hap','sad','neu'],
                   '5' : ['ang','hap','sad','neu', 'exc']}
        self.filelist = filelist
        self.sess_ids = sess_ids
        self.device = device
        df = pd.read_csv(filelist, names=['file', 'label', 'spk_id', 'gender'], sep=' ')
        files = df['file'].tolist()
        sessions = []
        for file in files:
            sessions.append(int(file.split("/", 1)[0][-1]))
        df = df.assign(sess_id=sessions)    
        df = df[(df['sess_id'].isin(sess_ids))]
        self.rootdir = rootdir
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.max_frm = int(max_len/160)          # hop length = 160
        self.n_classes = n_classes
        self.df = df[(df['label'].isin(emo_set[str(n_classes)]))]
        self.labels = dict()
        for i, emo in enumerate(emo_set[str(n_classes)]):
            self.labels[emo] = i

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        label = self.labels[row['label']]
        mfccfile = self.rootdir + '/' + row['file'].replace('/wav/', '/mfc/') + '.npy'
        mfcc = np.load(mfccfile)
        self.n_mfcc, cur_frm = mfcc.shape

        # Data loader only supports items of the same size unless we provide our own collate_fn()
        if cur_frm > self.max_frm:
            out_mfcc = mfcc[:, 0:self.max_frm]      # Trim the tail
        else:
            out_mfcc = np.zeros((self.n_mfcc, self.max_frm))     # Pad tail with 0     
            n_repeats = int(self.max_frm / cur_frm) # Repeat the waveform N times to fill the array
            for k in range(0, n_repeats):
                out_mfcc[:, k*cur_frm : (k+1)*cur_frm] = mfcc

        mfcc = torch.from_numpy(np.array(out_mfcc, dtype=np.float32)).to(self.device)
        label = np.array(label, dtype=np.float32)
        label = torch.tensor(np.array(label, dtype=np.float32)).to(self.device)
        return mfcc, label