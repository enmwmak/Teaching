import librosa
import numpy as np
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavfile', default='../../IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav')
    parser.add_argument('--mfcfile', default='../../IEMOCAP_full_release/Session1/sentences/mfc/Ses01F_impro01/Ses01F_impro01_F000.npy')

    args = parser.parse_args()
    wavfile = args.wavfile
 
    srate=librosa.get_samplerate(wavfile)
    plt.figure(figsize=(14, 9))
    plt.figure(1)
    
    plt.subplot(211)
    plt.title('MFCC')
    mfcc = np.load(args.mfcfile)
    librosa.display.specshow(librosa.amplitude_to_db(mfcc), sr=srate,
                             y_axis='off', hop_length=160)

    plt.subplot(212)
    plt.title('Waveform')
    waveform, srate = librosa.load(wavfile, sr=librosa.get_samplerate(wavfile))
    librosa.display.waveshow(waveform, sr=srate, offset=0)

    plt.margins(x=0)
    plt.show()