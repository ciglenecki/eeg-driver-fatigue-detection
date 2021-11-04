# Driver fatigue detection through multiple entropy fusion analysis in an EEG-based system

Paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188756

<center>

![](https://journals.plos.org/plosone/article/figure/image?size=inline&id=info:doi/10.1371/journal.pone.0188756.g002)

</center>



# Requirements

[./requirements.txt](./requirements.txt)

# Questions:

Should I implement all classifiers or only SVM/BP?

Should I recalculate significant electrodes or caculate them again?

How is autoregression used in this paper?



# Code notes
`ipython kernel install --user --name=eeg` to use venv

# Data notes

EEG data: This is the original EEG data of twelve healthy subjects for driver fatigue detection. Due to personal privacy, the digital number represents different participants. The .cnt files were created by a 40-channel Neuroscan amplifier, including the EEG data in two states in the process of driving.

Entropy data: This is the four entropies of twelve healthy subjects for driver fatigue detection. Due to personal privacy, the digital number represents different participants. Each .mat file included five files, namely FE, SE, AE and PE described four entropy values in the training data and the corresponding label of class, which the number 1 represents the fatigue state and 0 represents the normal state in the process of driving.

# Paper notes
## Goal
analyze the multiple entropy fusion method and evaluate several channel regions to effectively detect a driver’s fatigue state based on electroencephalogram (EEG) records


## Data:
- colleted by attaching electrodes to driver’s
- non-fatigue data: driver was driving for 20 minutes. Last 5 minutes are captured as non-fatigue
- fatigue data: driver was driving for 40-60 minutes. Last 5 minutes are captured as fatigue data. 

## Electorde cap:
- 32 channels (30 effective adn 2 reference channels)


## Entropies:
- PE - special entropy - calculated by applying the Shannon function to the normalized power spectrum based on the peaks of a Fourier transform
- AE - Approximate entropy - calculated in time domain without phase-space reconstruction of signal (short-length time series data) [41]
- SE - Sample entropy - similar to AE. Se is less sensitive to changes in data length with largers values corresponding to greater complexity or irregularity in the data [41]
- FE - Fuzzy entrop - stable results for different parameters. Best noise resistance using fuzzy membership function.

### Entropy Parameters (AE, SE, FE):
- m: dimension of phase space
	- m = 2
- r: similarity tolerance
	- r = 0.2 * SD (SD = standard deviation of the time series)

### Feature normalization
Features were normalized to [-1, 1] using min-max normalization:
1. Feature vector is built using the concatenation process, which concatenates the features.
2. The min-max normalization of each feature xi, i = 1,. . .,n, is computed as follows:


## 4 classifiers
1. Support vector machine (SVM)
2. Back propagation neural network (BP)
3. Random forest (RF)
4. K-nearest neighbor (KNN)

## SVM Parameters
With leave-one-out (LOO) cross-validation parameters :
1. c=-1 - the penality parameter
2. g=-5 - the kernel parameter
3. AR order 10.

## Entropy combining
Combining multiple entropies always yields better accuracy.

## Significant electores

Significant electores were chosen from 30 electores.
1. Calculate Acc(i) of single i electorde using multiple entropy fusion method based on training data by SVM classifier
2. Obtain accuracy for each electrode and then recalculate it by combining pairwise electrode (with 29 electrodes)
3. Calculate the weight V_i = Acc(i) + sum[ Acc(ij) + Acc(i) - Acc(j) ] / 30

Pick 10 electores with biggest weight. These 10 electordes produce 4 clusters/regions A,B,C,D. A gives the best prediction results and even better prediction compared when all electrodes were used for a prediction. 
