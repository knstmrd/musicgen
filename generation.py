import numpy as np
import feature_extraction as fp


def train_single(X, y, predictor):
    predictor.fit(X, y)
    return predictor


def generate_frame_single(features, predictor):
    return predictor.predict(features)


def generate(spectrogram, predictor, config, n_features):
    y = np.zeros((config['n_test'] + config['seed_length'],n_features))
    y[:config['seed_length'], :] = spectrogram[config['test_offset']:config['test_offset'] + config['seed_length'], :]
    for i in range(config['n_test']):
        if i % config['gen_verbose'] == 0:
            print('Generating frame {}/{}'.format(i, config['n_test']))
        features = fp.extract_features_single_frame(y, config['seed_length'] + i, config).reshape(1, -1)
        y[config['seed_length'] + i, :] = generate_frame_single(features, predictor)
    return y
