# Driver fatigue detection through multiple entropy fusion analysis in an EEG-based system

<p align="center">
	<img src="pics/header_image.png"></img>
</p>
<p align="center">
	<a align="center" href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188756">Paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188756</a>
</p>

# Requirements

[requirements.txt](requirements.txt)


# Notes to self:
- preprocessor, only preprocesses
- feature extract fit -> caculates psds


1. Many channels are flatlined during the driving process and they spike only in some moments
2. In addition each BCIT dataset includes 4 additional EOG channels placed vertically above the right eye (veou), vertically below the right eye (veol), horizontally on the outside of the right eye (heor), and horizontally on the outside of the left eye (heol)
3. `ipython kernel install --user --name=eeg` to use venv in jupyter
4. Normalized values produce nan for SE entropy (minmax scaler 1d)
5. Two different libs (EntropyHub and Antropy) produce the same result for sample entropy
6. Applying filter before converting to epochs and after is not the same
# Todo:
- [ ] Fix data leakage. Don't scale on the whole dataset. Scale only the train dataset seperatly of test data. Fit on train, transform on train, transform on test
- [ ] Instead of remove main features, append alpha beta gamma delta features along side the main ones
- [ ] Explore more options for feature extraction https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html
- [ ] Rereferencing - append columns instead of removing them
- [ ] Explore which features are most important for prediction
### Utils:
- [x] Create report file saver and loader for easy and reproducible way to check results

### Signal:
- [x] Apply filters to remove noise
	- [x] notch filter 50Hz
	- [x] band pass 0.15Hz to 40Hz
- [x] Crop the signal to 5 minutes (300 seconds) 
- [x] Load signal for all users
- [x] Create epoch from the signal using the window of 1 second
- [x] Calculate 4 different entropies for each 1 second epoch

### Dataframe:
- [x] Concatenate features into a final dataframe
- [x] Normalize across all participants and not single one
- [x] Filter bad values and replace them with 0

### Train:
- [x] Split the dataset to train and test set (1:1)
- [x] Use LOO (leave one participant out) approach to find the best `C` and `gamma` parameters for the SVM model
- [x] Train the SVM model with multiple combinations of entropies (function `powerset`) to find out which entropy combination has the highest accuracy on the train dataset 
- [x] Train the following models using the Grid Search method:
	- [x] SVM
	- [x] Neural network (BP)
	- [x] KNN
	- [x] Random Forest (RF)
- [x] Validate accuracy using testing set and report performance on each model
- [x] Determine significant electrodes by calculating the weight for each electrode for each user with the formula describe in the research paper:
	- $$V_i=\frac{Acc(i) + \sum_{j=1, j\not=i}^{30}{Acc_{(ij)} + Acc_{(i)} - Acc_{(j)}}}{30}$$

Improvement:
- [x] Check SE entropy infs
  - There are no inf and NaN values anymore once this was fixed 
- [x] Alpha beta gama delta waves
- [x] Additional features, mean, psd
- [x] Training with additional features
- [x] Training with additional features and brainwave bands
- [x] ICA - Principal component analysis
	- [x] filter low 1hz to remove drifts	

Optional:
- [ ] Repeat training with significant electrodes
- [ ] Compare entropies with entropies from the paper
- [ ] Visualize training/testing error
- [ ] Visualize weight-based topographies for each subject
- [ ] Visualize weight-based topographies average



# Questions:

How is AR (auto-regression) is often mention in the paper. What it's use in the context of the problem?

# Dataframe structure

Here, we will calculate the entropy (4) for every channel (30) for every epoch. In the research paper, they also did that but reduced number of entropies from (30 * 4) to (4) by doing a "a feature-level fusion"

| user_id | epoch_id | label | PE_CH01 | PE_CH02 | ... | PE_CH30 | SE_CH01 | SE_CH02 | ... | FE_CH30 |
| ------- | -------- | ----- | ------- | ------- | --- | ------- | ------- | ------- | --- | ------- |
| 01      | 0        | 0     | 0.3     | 0.23    | ... | 0.6     | 0.8     | 0.1     | ... | 0.2     |
| 01      | 1        | 0     | 0.2     | 0.1     | ... | 0       | 0.2     | 0.1     | ... | 0.2     |
| ...     | ...      | ...   | ...     | ...     | ... | ...     | ...     | ...     | ... | ...     |
| 01      | 0        | 0     | 0.6     | 0.3     | ... | 0.1     | 0.2     | 0.5     | ... | 0.1     |
| 02      | 1        | 0     | 0.2     | 0.1     | ... | 0       | 0.2     | 0.1     | ... | 0.2     |
| ...     | ...      | ...   | ...     | ...     | ... | ...     | ...     | ...     | ... | ...     |

Number of rows: 

```
users (12) * epochs (300) * driving_states (2) = 7200
```

Number of columns:
```
user_id (1) + label (1) + epoch_id (1) + entropies (4) * channels (30) = 123
```



# Dataset notes

EEG data:
- .cnt files were created by a 40-channel Neuroscan amplifier including the EEG data in two driving_states in the process of driving.

Entropy data:
- four entropies of twelve healthy subjects for driver fatigue detection
- the digital number represents different participants
- each .mat file included five files
	- FE
	- SE
	- AE
	- PE described four entropy values in the training data
	- Class_label 0 or 1
		- 1 represents the fatigue state
		- 0 represents the normal state

# Reserach paper notes
## Goal
analyze the multiple entropy fusion method and evaluate several channel regions to effectively detect a driver’s fatigue state based on electroencephalogram (EEG) records


## Data:
- collected by attaching electrodes to driver’s
- non-fatigue data: driver was driving for 20 minutes. Last 5 minutes are captured as non-fatigue
- fatigue data: driver was driving for 40-60 minutes. Last 5 minutes are captured as fatigue data. 
- dataset is split randomly 50:50 train/test
- 5 minute EEG data from 30 electrodes
	- sectioned into 1 second epoch
	- 5 * 60 = 300 * 1 = 300 epoch for one participant
	- total 3600 fatigue units and 3600 normal units 

## Electrode cap:
- 32 channels (30 effective and 2 reference channels)


## Entropies:
- PE - special entropy - calculated by applying the Shannon function to the normalized power spectrum based on the peaks of a Fourier transform
- AE - Approximate entropy - calculated in time domain without phase-space reconstruction of signal (short-length time series data) [41]
- SE - Sample entropy - similar to AE. Se is less sensitive to changes in data length with larger values corresponding to greater complexity or irregularity in the data [41]
- FE - Fuzzy entropy - stable results for different parameters. Best noise resistance using fuzzy membership function.

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
1. c=-1 - the penalty parameter
2. g=-5 - the kernel parameter
3. AR order 10.

## Entropy combining
Combining multiple entropies always yields better accuracy.

## Significant electrodes

Significant electrodes were chosen from 30 electrodes.
1. Calculate Acc(i) of single i electrode using multiple entropy fusion method based on training data by SVM classifier
2. Obtain accuracy for each electrode and then recalculate it by combining pairwise electrode (with 29 electrodes)
3. Calculate the weight for each electrode $V_i=\frac{Acc(i) + \sum_{j=1, j\not=i}^{30}{Acc_{(ij)} + Acc_{(i)} - Acc_{(j)}}}{30}$

Pick 10 electrodes with biggest weight. These 10 electrodes produce 4 clusters/regions A,B,C,D.
- A gives the best prediction results and even better prediction compared when all electrodes were used for a prediction