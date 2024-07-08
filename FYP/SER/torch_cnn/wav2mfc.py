'''
Convert all .wav files (sentences) in IEMOCAP to .mfc files according to the filelist 
'emo_labels_cat.txt'. The program will add a folder (mfc/) under 
IEMOCAP_full_release/Sessionx/sentences and save the .mfc files there.
'''

import pandas as pd
import librosa
import numpy as np
import os

rootdir = '../../IEMOCAP_full_release'
filelist = '../../labels/emo_labels_cat.txt'
n_mfcc = 20

df = pd.read_csv(filelist, names=['file', 'label', 'spk_id', 'gender'], sep=' ')
for i in range(0, len(df)):
    row = df.iloc[i]
    wavfile = rootdir + '/' + row['file'] + '.wav'
    mfcfile = rootdir + '/' + row['file'].replace('/wav/', '/mfc/') + '.npy'
    print(f'{wavfile} --> {mfcfile}')

    waveform, srate = librosa.load(wavfile)
    mfcc = librosa.feature.mfcc(y=waveform, sr=srate, n_mfcc=n_mfcc,
                                n_fft=512, hop_length=160, center=False)
    dmfcc = librosa.feature.delta(mfcc)
    ddmfcc = librosa.feature.delta(dmfcc)
    features = np.vstack((mfcc, dmfcc, ddmfcc))
    
    os.makedirs(os.path.dirname(mfcfile), exist_ok=True)
    np.save(mfcfile, features)
