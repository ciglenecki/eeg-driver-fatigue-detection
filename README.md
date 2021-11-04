# Driver fatigue detection through multiple entropy fusion analysis in an EEG-based system

Paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188756

# Requirements
```
sklearn
pandas
mne
numpy
```

# Questions:

Do vašeg termina konzultacija morate napraviti sljedeće:

    GitHub repozitorij i mene dodati u njega (username je IgorStancin)
    Pročitati s razumijevanjem članak koji ste izabrali (preporučam pročitati nekoliko puta)
    Preuzeti podatke koji su korišteni u članku
    Napraviti „data survey“ Jupyter bilježnicu u kojoj ćete se upoznati s podatcima kojim baratate (tip podataka, distribucije, nedostajuće vrijednosti, stršeće vrijednosti, razne vizualizacije i slično)
    Pripremiti kratki plan kako planirate replicirati odabrani članak (maksimalno 1 stranica teksta/natuknica) – iz teksta članka prepoznati sve bitne dijelove, odrediti redoslijed kojim ćete ih odrađivati, koje Python pakete ćete koristiti za koju potkomponentu i slično.

 

Na terminu konzultacija čete mi kratko prikazati najzanimljivije stvari iz navedene bilježnice i plan koji ste napravili. Sama bilježnica mora biti detaljna i cjelovita, ali prezentacija bilježnice mora biti samo za najzanimljivije stvari koje ste uočili u podatcima i ne smije biti duža od 3 minute. Prezentirani plan ćemo kratko prokomentirati i ja ću vam odgovoriti na sva vaša eventualna pitanja. Predviđeno trajanje konzultacija je maksimalno 15 minuta po studentu.




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