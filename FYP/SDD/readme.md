This folder contains examples of Python programs (called by run.sh) for speech depression detection (SDD) on the DAIC-WOZ corpus. 
The program "sdd.py" performs leave-one-speaker-out cross-validation on the training set and evaluates the CNN model's performance on
the test set of the corpus. The programs use a small 1D-CNN with statistics pooling or attentive statistics pooling to minimize computation. 
Therefore, they only serve as examples and a coding framework for students to learn CNN and cross-validation. 
To improve performance, you may replace the 1D-CNN with more advanced networks such as ConFormer. Also, 
you may replace the MFCC extraction (in dataset.py) with wav2vec2, HuBERT, or WavLM features.

The programs should be run under the "programs/torch_cnn" folder. The DAIC-WOZ dataset should be put under the 
"DAIC-WOZ/" folder. All label files should be put inside the "labels/" folder.

Read the following papers for more advanced methods:

- Lishi Zuo and Man-Wai Mak, "Avoiding Dominance of Speaker Features in Speech-based Depression Detection", Pattern Recognition Letters, vol. 173, Sep 2023, pp. 50-56. 
- Lishi Zuo, Man-Wai Mak, and Youzhi TU, "Promoting Independence of Depression and Speaker Features for Speaker Disentanglement in Speech-Based Depression Detection", ICASSP, Seoul, April, 2024, pp. 10191-10195.

