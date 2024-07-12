#!/bin/sh -ex
# Master script running programs in sequence. Run this scrip under program/torch_cnn/ directory
# To execute this script, type
#   ./run.sh <stage>

# Get stage from command line argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <stage>"
fi
stage=$1

#=======================================================
# Stage 1: Prepare data
#=======================================================
if [ $stage -eq 1 ]; then
    mkdir -p ../../labels
    rm -rf ../../audio/wav
    python3 extract_wav.py --prot_file ../../DAIC-WOZ/protocol/train_split_Depression_AVEC2017.csv \
        --etype train --wtype segs --corpus_dir ../../DAIC-WOZ > ../../labels/ssd_labels_segs_train.txt 
    python3 extract_wav.py --prot_file ../../DAIC-WOZ/protocol/full_test_split.csv \
        --etype test --wtype segs --corpus_dir ../../DAIC-WOZ > ../../labels/ssd_labels_segs_test.txt
    python3 extract_wav.py --prot_file ../../DAIC-WOZ/protocol/dev_split_Depression_AVEC2017.csv \
        --etype dev --wtype segs --corpus_dir ../../DAIC-WOZ > ../../labels/ssd_labels_segs_dev.txt 
fi

#=======================================================
# Stage 2: Convert wav to mfcc
#=======================================================
if [ $stage -eq 2 ]; then
    rm -rf ../../audio/mfc
    python3 wav2mfc.py
fi

#=======================================================
# Stage 3: Extract COVAREP features
#=======================================================
if [ $stage -eq 3 ]; then
    rm -rf ../../audio/cov
    python3 extract_cov.py --prot_file ../../DAIC-WOZ/protocol/train_split_Depression_AVEC2017.csv \
                --etype train --wtype segs --corpus_dir ../../DAIC-WOZ 
    python3 extract_cov.py --prot_file ../../DAIC-WOZ/protocol/full_test_split.csv \
                --etype test --wtype segs --corpus_dir ../../DAIC-WOZ 
    python3 extract_cov.py --prot_file ../../DAIC-WOZ/protocol/dev_split_Depression_AVEC2017.csv \
                --etype dev --wtype segs --corpus_dir ../../DAIC-WOZ 
fi

#=======================================================
# Stage 4: Extract mel-spectrograms
#=======================================================
if [ $stage -eq 4 ]; then
    rm -rf ../../audio/mel
    python3 wav2mel.py
fi

#=======================================================
# Stage 5: Train and evaluate the CNN
#=======================================================
if [ $stage -eq 5 ]; then
    python3 sdd.py --wtype segs --pool_method sp --model_file models/sdd_cnn_stats.pth \
        --trainlist ../../labels/ssd_labels_segs_train.txt \
        --testlist ../../labels/ssd_labels_segs_test.txt \
        --n_epochs 10 --audiodir ../../audio --ftype mfc
fi


