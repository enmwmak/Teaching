This folder contains examples of Python programs for speech emotion recognition (SER) on the IEMOCAP corpus. The programs perform leave-one-session-out (ser_session.py) and leave-one-speaker-out (ser_speaker.py) cross-validation. The programs use a small 1D-CNN with statistics pooling or attentive statistics pooling to minimize computation. Therefore, they only serve as examples and a coding framework for students to learn CNN and cross-validation. To improve performance, you may replace the 1D-CNN with more advanced networks such as ConFormer. Also, you may replace the MFCC extraction (in dataset.py) with wav2vec2, HuBERT, or WavLM features.

The programs should be run under the "programs/torch_cnn" folder. The IEMOCAP dataset should be put under the "IEMOCAP_full_release/" folder. All label files should be put inside the "labels/" folder.

The default settings are as follows:
- No. of class: 4 (1103 'ang', 1636 'hap+exc', 1084 sad, 1708 neu)
- No. of samples: 5531
- Label file: emo_labels_cat_exc-as-hap.txt
