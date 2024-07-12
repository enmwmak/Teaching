'''
Convert all .wav files in audio/wav to mel-spectrogram files (in .npy format) according to a filelist 
the ../../labels directory. The program will add a folder (../../audio/mfc/) 
'''

import pandas as pd
import librosa
import numpy as np
import os

def wav2mel(filelist, audiodir='../../audio', n_fft=512):
    df = pd.read_csv(filelist, names=['file', 'label', 'spk_id', 'gender'], sep=' ')
    wavdir = audiodir + '/wav'
    meldir = audiodir + '/mel'
    for i in range(0, len(df)):
        row = df.iloc[i]
        wavfile = wavdir + '/' + row['file'] + '.wav'
        melfile = meldir + '/' + row['file'] + '.npy'
        print(f'{wavfile} --> {melfile}')

        waveform, srate = librosa.load(wavfile, sr=librosa.get_samplerate(wavfile))
        n_samples = len(waveform)
        n_fft = n_samples if (n_samples < 512) else 512
        melspec = librosa.feature.melspectrogram(y=waveform, sr=srate, n_fft=n_fft, 
                                              hop_length=160, center=False)
        os.makedirs(os.path.dirname(melfile), exist_ok=True)
        np.save(melfile, melspec)


if __name__ == "__main__":
    wav2mel('../../labels/ssd_labels_segs_train.txt')
    wav2mel('../../labels/ssd_labels_segs_dev.txt')
    wav2mel('../../labels/ssd_labels_segs_test.txt')

