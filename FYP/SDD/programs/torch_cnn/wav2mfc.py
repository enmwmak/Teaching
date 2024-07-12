'''
Convert all .wav files in audio/wav to MFCC files (in .npy format) according to a filelist 
the ../../labels directory. The program will add a folder (../../audio/mfc/) 
'''

import pandas as pd
import librosa
import numpy as np
import os

def wav2mfc(filelist, audiodir='../../audio', n_mfcc=20, n_fft=512):
    df = pd.read_csv(filelist, names=['file', 'label', 'spk_id', 'gender'], sep=' ')
    wavdir = audiodir + '/wav'
    mfcdir = audiodir + '/mfc'
    for i in range(0, len(df)):
        row = df.iloc[i]
        wavfile = wavdir + '/' + row['file'] + '.wav'
        mfcfile = mfcdir + '/' + row['file'] + '.npy'
        print(f'{wavfile} --> {mfcfile}')

        waveform, srate = librosa.load(wavfile, sr=librosa.get_samplerate(wavfile))
        n_samples = len(waveform)
        n_fft = n_samples if (n_samples < 512) else 512
        mfcc = librosa.feature.mfcc(y=waveform, sr=srate, n_mfcc=n_mfcc,
                                    n_fft=n_fft, hop_length=160, center=False)
        n_frames = mfcc.shape[1]
        if n_frames > 9:
            dmfcc = librosa.feature.delta(mfcc)
            ddmfcc = librosa.feature.delta(dmfcc)
        else:
            dmfcc = np.zeros(mfcc.shape)
            ddmfcc = np.zeros(mfcc.shape)    
        features = np.vstack((mfcc, dmfcc, ddmfcc))

        os.makedirs(os.path.dirname(mfcfile), exist_ok=True)
        np.save(mfcfile, features)


if __name__ == "__main__":
    wav2mfc('../../labels/ssd_labels_segs_train.txt')
    wav2mfc('../../labels/ssd_labels_segs_dev.txt')
    wav2mfc('../../labels/ssd_labels_segs_test.txt')

