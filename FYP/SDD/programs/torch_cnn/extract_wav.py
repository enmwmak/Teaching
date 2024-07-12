'''
- Extract the speech segments from the subjects in DAIC-WOZ based on the time stamps in the 
transcription files. For each recording session, the program concatenates the participant's speech
into a wav file in the folder audio/wav. It also partition the participant's speech 
into 3.84-second segments, following the paper:
    DepAudioNet: An Efficient Deep Model for Audio based Depression Classification

The program also outputs a list with the following contents 
(similar to those in iemocap/labels/emo_labels_cat.txt) to stdout.
    <wav file w/o .wav extention> <class label> <speaker id> <gender>
    e.g.,
    303/303_0 nor 303 F
    ...
    459/459_82 dep 459 M                 
'''

import pandas as pd
import librosa
import numpy as np
import os
import soundfile as sf
import argparse

def extract_segs_from_wav(wavfile, tranfile, spkinfo, seg_dur=3.84):
    '''
    Each segments (3.84s) is saved as a 16kHz 16bit wav file
    '''
    tgt_sr = 16000
    waveform, srate = librosa.load(wavfile)

    # Convert to 16kHz 16bit/sample .wav file
    waveform = librosa.resample(waveform, orig_sr=srate, target_sr=tgt_sr)

    # Speaker ID is encoded in the wavfile name before the '_' char
    spkid = wavfile.split('/')[-1].split('_')[0]

    # Dataframes for labels and transcriptions. Consider the participant only
    lb_df = pd.DataFrame(columns=['segment', 'label', 'speaker', 'gender'])
    tr_df = pd.read_csv(tranfile, names=['start_time', 'stop_time', 'speaker value', 'text'], sep='\t')
    tr_df = tr_df[(tr_df['speaker value'].isin(['Participant']))]

    session = np.empty((0))                              # Storing all utterances of the participant
    for i in range(0, len(tr_df)):
        row = tr_df.iloc[i]
        start_smp = int(float(row['start_time']) * tgt_sr)
        stop_smp = int(float(row['stop_time']) * tgt_sr)
        response = waveform[start_smp : stop_smp]       # Extract response of participant
        session = np.hstack((session, response))        # Concate waveform

    # Collect gender and label information
    gender_val = spkinfo['Gender']
    gender = 'M' if gender_val == 1 else 'F'
    label_val = spkinfo['Label value']
    label = 'dep' if label_val == 1 else 'nor'

    # Partition the whole session into 3.84-sec segments and save each segment to a wav file
    n_smps = int(seg_dur * tgt_sr)            # No. of samples in each segment
    n_segs = int(session.shape[0]/n_smps)    
    for s in range(0, n_segs):
        segment = session[s*n_smps : (s+1)*n_smps]
        seg_wavfile = wavfile.replace('DAIC-WOZ/', 'audio/wav/').replace('AUDIO', str(s))
        os.makedirs(os.path.dirname(seg_wavfile), exist_ok=True)
        sf.write(seg_wavfile, segment, samplerate=tgt_sr, subtype='PCM_16')
        seg_name = seg_wavfile.replace('.wav', '').split('wav/')[-1]
        lb_df.loc[s] = [seg_name, label, spkid, gender] 

    # Output label info to stdout
    out = lb_df.to_csv(None, index=False, sep=" ", header=False)
    out = out[0:-1]     # Remove the last '\n'
    print(out)


def extract_sess_from_wav(wavfile, tranfile, spkinfo):
    '''
    The utterances of the participant in a session are concatenated. The concatenated 
    waveform saved as a 16kHz 16bit wav file
    '''
    waveform, srate = librosa.load(wavfile)
    tgt_sr = 16000

    # Convert to 16kHz 16bit/sample .wav file
    waveform = librosa.resample(waveform, orig_sr=srate, target_sr=tgt_sr)

    # Speaker ID is encoded in the wavfile name before the '_' char
    spkid = wavfile.split('/')[-1].split('_')[0]

    # Dataframes for labels and transcriptions. Consider the participant only
    lb_df = pd.DataFrame(columns=['segment', 'label', 'speaker', 'gender'])
    tr_df = pd.read_csv(tranfile, names=['start_time', 'stop_time', 'speaker value', 'text'], sep='\t')
    tr_df = tr_df[(tr_df['speaker value'].isin(['Participant']))]

    session = np.empty((0))                              # Storing all utterances of the participant
    for i in range(0, len(tr_df)):
        row = tr_df.iloc[i]
        start_smp = int(float(row['start_time']) * tgt_sr)
        stop_smp = int(float(row['stop_time']) * tgt_sr)
        response = waveform[start_smp : stop_smp]       # Extract response of participant
        session = np.hstack((session, response))        # Concate waveform

    # Save the concatenated wavform to 16kHz 16bit wav file
    sess_wavfile = wavfile.replace('DAIC-WOZ/', 'audio/wav/').replace('_AUDIO', '_sess')
    os.makedirs(os.path.dirname(sess_wavfile), exist_ok=True)
    sf.write(sess_wavfile, session, samplerate=tgt_sr, subtype='PCM_16')

    # Collect label info
    gender_val = spkinfo['Gender']
    gender = 'M' if gender_val == 1 else 'F'
    label_val = spkinfo['Label value']
    label = 'dep' if label_val == 1 else 'nor'
    sess_name = sess_wavfile.replace('.wav', '').split('wav/')[-1]
    lb_df.loc[i] = [sess_name, label, spkid, gender] 

    # Output label info to stdout
    out = lb_df.to_csv(None, index=False, sep=" ", header=False)
    out = out[0:-1]     # Remove the last '\n'
    print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prot_file', default='../../DAIC-WOZ/protocol/full_test_split.csv')
    parser.add_argument('--etype', choices=['train', 'test', 'dev'], default='test')
    parser.add_argument('--wtype', choices=['segs', 'sess'], default='sess')
    parser.add_argument('--corpus_dir', default='../../DAIC-WOZ')
    args = parser.parse_args()
    corpus_dir = args.corpus_dir
    prot_file = args.prot_file

    if (args.etype == 'train' or args.etype == 'dev'):
        df = pd.read_csv(prot_file, usecols=['Participant_ID','PHQ8_Binary','Gender'], sep=',')
        df = df.rename(columns={'PHQ8_Binary': 'Label value'})
    else:
        df = pd.read_csv(prot_file, usecols=['Participant_ID','PHQ_Binary','Gender'], sep=',')
        df = df.rename(columns={'PHQ_Binary': 'Label value'})
            
    for i in range(0, len(df)):
        spkinfo = df.iloc[i]
        spkid = str(spkinfo['Participant_ID'])
        wavfile = corpus_dir + '/' + spkid + '/' + spkid + '_AUDIO.wav'
        tranfile = corpus_dir + '/' + spkid + '/' + spkid + '_TRANSCRIPT.csv'
        if args.wtype == 'segs':
            extract_segs_from_wav(wavfile, tranfile, spkinfo, seg_dur=3.84)
        else:
            extract_sess_from_wav(wavfile, tranfile, spkinfo)    
