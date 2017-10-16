# Musicgen

Raw audio generation experiments.

Basic idea: compute the STFT of some audio, extract features, generate new STFT step-by-step (extracting new features during each step). Currently replacing phase data with random noise and only predicting amplitudes, which leads to everything sounding pretty bad. The feature extraction has not been optimized (at all), so it takes up a lot of the 

## Logistic Regression

[https://github.com/knstmrd/musicgen/blob/master/music-generation-v2.ipynb](Audio generation using logistic regression) - there a lot of bugs, but since this method does not produce very good results in general, I've abandoned its development

The good:

* fast training and generation

The bad:

* can "blow up" (amplitude goes to infinity), attempts to set a maximum volume lead to unpleasant clicks and frequent "resetting" of the values.

## Decision Trees, Boosted Trees and Random Forests

[https://github.com/knstmrd/musicgen/blob/master/music-generation-trees.ipynb](Audio generation using decision trees, but any regressor can be used)

Trees take a lot of time to train, and training multi-output trees gives a steady droning output which is less interesting than the results given by separate trees (but those take a large amount of time to train)

The good:

* More interesting results

The bad:

* Excruciatingly slow

* Still end up giving drones

# Future plans

* Test adding random to features noise during generation to produce more varied audio

* Test various neural networks

* Test audio-specific features

* Speed up feature extraction

* Try some sort of Phase Vocoder approach to get rid of phase issues (akin to [http://www.johnglover.net/blog/generating-sound-with-rnns.html](this approach))