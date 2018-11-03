# Musicgen

Raw audio generation experiments.

## New code (2018)

Basic idea: compute mel-spectrogram of audio, pre-process, generate new mel-spectrograms step-by-step, apply inverse mel-transform, apply Griffin-Lim algorithm to estimate phase of audio.

General to-do:

- [ ] Apply LeRoux's reconstruction algorithm

### Linear regression, various tree algorithms, vanilla neural networks

Idea: extract features, feed to regression algorithm, generate new audio frame.

### Neural network based approaches

Idea: feed raw audio frame(s) to neural network.

Approaches:

- [x] RNN-based generation (currently via one-layer LSTMs)
- [ ] CNN-based generation
- [ ] GAN-based generation (combined with CNNs?)
- [ ] Other approaches (convolutional RNNs?)

### TO-DO list

- [ ] Pre-compute and store mel-spectrograms, since otherwise it takes time to load audio and perform STFT

## Old notebooks (2017)

Approaches tried in 2017, may contain bugs, etc.

Basic idea: compute the STFT of some audio, extract features, generate new STFT step-by-step (extracting new features during each step). Currently replacing phase data with random noise and only predicting amplitudes, which leads to everything sounding pretty bad. The feature extraction has not been optimized (at all), so it takes up a lot of the computation time.

### Logistic Regression

[Audio generation using logistic regression](https://github.com/knstmrd/musicgen/blob/master/music-generation-v2.ipynb) - there's a lot of bugs, but since this method does not produce very good results in general, I've abandoned its development

The good:

* fast training and generation

The bad:

* can "blow up" (amplitude goes to infinity), attempts to set a maximum volume lead to unpleasant clicks and frequent "resetting" of the values.

### Decision Trees, Boosted Trees and Random Forests

[Audio generation using decision trees, but any regressor can be used](https://github.com/knstmrd/musicgen/blob/master/music-generation-trees.ipynb)

Trees take a lot of time to train, and training multi-output trees gives a steady droning output which is less interesting than the results given by separate trees (but those take a large amount of time to train). This version can pre-load features from a file or write extracted features to a file.

The good:

* More interesting results

The bad:

* Excruciatingly slow

* Still end up giving drones