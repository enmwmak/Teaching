'''
- Extract the COVAREP feature segments from the subjects in DAIC-WOZ based on the time stamps in the 
transcription files. For each recording session, the program concatenates the participant's COVAREP
vectors into one numpy array. Then, it partitions the participant's COVAREP feature vectors into 
3.84-second segments, following the paper:
    DepAudioNet: An Efficient Deep Model for Audio based Depression Classification
'''

import pandas as pd
import numpy as np
import os
import argparse

def extract_segs_from_cov(covfile, tranfile, spkinfo, seg_dur=3.84):
    '''
    Each segments (3.84s) is saved as an .npy file
    '''
    covfull = np.genfromtxt(covfile, delimiter=',')
    dim = covfull.shape[1]         # Dim of COVAREP features

    # Speaker ID is encoded in the covfile name before the '_' char
    spkid = covfile.split('/')[-1].split('_')[0]
    frm_rate = 100          # DAIC-WOZ uses 100Hz frame rate

    # Dataframes for labels and transcriptions. Consider the participant only
    tr_df = pd.read_csv(tranfile, names=['start_time', 'stop_time', 'speaker value', 'text'], sep='\t')
    tr_df = tr_df[(tr_df['speaker value'].isin(['Participant']))]

    session = np.empty((0, dim))                              # Storing all cov of the participant
    for i in range(0, len(tr_df)):
        row = tr_df.iloc[i]
        start_frm = int(float(row['start_time'])*frm_rate)
        stop_frm = int(float(row['stop_time'])*frm_rate)
        response = covfull[start_frm : stop_frm][:]       # Extract response of participant
        session = np.vstack((session, response))

    # Partition the whole session into 3.84-sec segments and save each segment to an .npy file
    n_frms = int(seg_dur * frm_rate)            # No. of frames in 3.84s. 
    n_segs = int(session.shape[0]/n_frms)
    for s in range(0, n_segs):
        segment = session[s*n_frms : (s+1)*n_frms][:]
        seg_covfile = covfile.replace('DAIC-WOZ/', 'audio/cov/').replace('COVAREP.csv', str(s)+'.npy')
        print(f'{covfile} --> {seg_covfile}')
        os.makedirs(os.path.dirname(seg_covfile), exist_ok=True)
        np.save(seg_covfile, np.transpose(segment))     # Consistent with MFC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prot_file', default='../../DAIC-WOZ/protocol/train_split_Depression_AVEC2017.csv')
    parser.add_argument('--etype', choices=['train', 'test', 'dev'], default='train')
    parser.add_argument('--wtype', choices=['segs', 'sess'], default='segs')
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
        covfile = corpus_dir + '/' + spkid + '/' + spkid + '_COVAREP.csv'
        tranfile = corpus_dir + '/' + spkid + '/' + spkid + '_TRANSCRIPT.csv'
        if args.wtype == 'segs':
            extract_segs_from_cov(covfile, tranfile, spkinfo)
        else:
            print('Extraction of session-based COVAREP features not implemented')
