import librosa
import numpy as np
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavfile', default='../../audio/wav/303/303_0.wav')
    parser.add_argument('--melfile', default='../../audio/mel/303/303_0.npy')
    parser.add_argument('--mfcfile', default='../../audio/mfc/303/303_0.npy')
    parser.add_argument('--covfile', default='../../audio/cov/303/303_0.npy')

    args = parser.parse_args()
    wavfile = args.wavfile
 
    srate=librosa.get_samplerate(wavfile)
    plt.figure(figsize=(14, 9))
    plt.figure(1)
    
    plt.subplot(411)
    plt.title('Mel-spectrogram')
    melspec = np.load(args.melfile)
    librosa.display.specshow(librosa.amplitude_to_db(melspec), sr=srate,
                             y_axis='linear', hop_length=160)

    plt.subplot(412)
    plt.title('MFCC')
    mfcc = np.load(args.mfcfile)
    librosa.display.specshow(librosa.amplitude_to_db(mfcc), sr=srate,
                             y_axis='off', hop_length=160)

    plt.subplot(413)
    plt.title('COVAREP')
    covr = np.load(args.covfile)
    librosa.display.specshow(librosa.amplitude_to_db(covr), sr=srate, hop_length=160)

    plt.subplot(414)
    plt.title('Waveform')
    waveform, srate = librosa.load(wavfile, sr=librosa.get_samplerate(wavfile))
    librosa.display.waveshow(waveform, sr=srate, offset=0)

    plt.margins(x=0)
    plt.show()