import pandas as pd
import numpy as np
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.callbacks as callbacks

def create_sequence(X, y, timesteps):
    """
    Creates sequences and targets (in a format that can be passed into a
    Keras RNN)

    :param X: LFP voltage readings, concatenated with timestep encodings

    :param y: binary data indicating whether a trial is a mirror or saccade

    :param timesteps: number of timesteps (in ms) of LFP voltage readings
    used for decoding

    :return: sequences: 3d Numpy array, each entry represents an LFP voltage sequence used for prediction

    targets: 1d numpy array, each entry is a binary bit indicating whether
    the trial is a saccade
    """
    sequences = []
    targets = []
    for i in range(X.shape[0] - timesteps + 1):
        a = X[i: i + timesteps, :]
        sequences.append(a)
    targets = y[timesteps - 1:]  
    return np.array(sequences), np.array(targets)


def create_trials(data, window_size, num_trials, steps_per_trial, timesteps,
                  center=0, timestep_encoding=False, bin_size=1):
    """
    Separates LFP voltage data into trials, in which the time point in the
    trial is encoded.

    :param data: DataFrame containing LFP voltage data, trial index,
    and whether the trial was a mirror or saccade

    :param window_size: amount of time (in ms) to use from each trial --
    each trial is 2000ms, which is more data than needed

    :param num_trials: number of trials in dataset

    :param steps_per_trial: amount of time (in ms) in trial

    :param timesteps: number of timesteps (in ms) of LFP voltage readings
    used for prediction

    :param center: where to center the trial (0 is time of eye fixation onset

    :param timestep_encoding: boolean indicating whether to encode the time
    point in the trial in the input data

    :param bin_size: indicates the amount of time (in ms) for each time
    point encoding. Only relevant if timestep_encoding is True. So,
    for example, if window_size = 700, and bin_size = 10, there are 700/10 = 70 possible
    encodings for a timestep

    :return:
    """
    X_trials = dict()
    y_trials = dict()
    midpoint = int(steps_per_trial/2) + center
    for i in range(num_trials):
        trial = data[data.trial_num == i]
        y_trial = np.array(trial.saccade)
        X_trial = np.array(trial.drop(columns=['saccade', 'trial_num']))
        X_trials[i] = X_trial[midpoint - int(window_size/2) - int(timesteps/2): midpoint + int(window_size/2) +
                              int(timesteps/2)]
        y_trials[i] = y_trial[midpoint - int(window_size/2) - int(timesteps/2): midpoint + int(window_size/2) +
                             int(timesteps/2)]
        if timestep_encoding == True:
            encoding = np.zeros((window_size + timesteps, int(window_size/bin_size)))
            k = 0
            j = 0
            while k < window_size:
                encoding[k:k+bin_size, j] = 1
                k += bin_size
                j += 1
            X_trials[i] = np.concatenate((X_trials[i], encoding), axis = 1)
    return X_trials, y_trials

def train_test_val_split(num_trials):
    """
     Separates trial indices into train, test, and validation indeces

    :param num_trials: number of total trials, across all eye directions

    :return: train_inds, test_inds, val_inds -- lists containing train,
    test, and validation indeces, respectively
    """
    inds = np.arange(num_trials)
    # set seed to ensure consistency
    np.random.seed(0)
    np.random.shuffle(inds)
    split = .2
    test_cutoff = int(num_trials * (1 - split))
    val_cutoff = int(test_cutoff * (1 - split))
    train_inds = inds[:val_cutoff]
    val_inds = inds[val_cutoff:test_cutoff]
    test_inds = inds[test_cutoff:]
    return train_inds, test_inds, val_inds

def normalize_trials(trials, train_inds, dimensions):
    """
    Normalizes all LFP voltages wrt to training data so that the LFP
    voltages have a mean of 0 and std of 1.

    :param trials: LFP voltage and (possibly timestep encoding),
    in dictionary format

    :param train_inds: trial indeces used for training the model

    :param dimensions: number of channels for LFP

    :return: trials: Normalized LFP voltage and (possibly timestep encoding) in
    dictionary format

    means, stds: mean and standard deviation of LFP voltage
    """
    train_data = dict()
    i = 0
    for ind in train_inds:
        if i == 0:
            train_data = trials[ind]
        else:
            train_data = np.concatenate((train_data, trials[i]))
    means = np.mean(train_data, axis=0)
    stds = np.std(train_data, axis=0)
    for key in trials:
        trials[key][:, :dimensions] = (trials[key][:, :dimensions, ] - means[
                                                                      :dimensions])/stds[:dimensions]
    return trials, means, stds


def create_sequences(X_trials, y_trials, train_inds, test_inds, val_inds, timesteps):
    """
    Creates train, test, and validation sequences for the entire dataset

    :param X_trials: dictionary containing LFP voltage trials

    :param y_trials: dictionary indicating whether trial is mirror or saccade

    :param train_inds: trial indeces used for training

    :param test_inds: trial indeces used for testing

    :param val_inds: trial indeces used for validation

    :param timesteps: amount of time (in ms) to look back at LFP voltage data

    :return: trial_sequences, trial_targets: dictionaries, where each entry
    is a sequence and target for a single trial

    X_train, y_train, X_test, y_test, X_val, y_val: train, test
    """
    trial_sequences = dict()
    trial_targets = dict()
    # fill in dictionaries containing sequences
    for key in X_trials.keys():
        sequence, target = create_sequence(X_trials[key], y_trials[key], timesteps)
        trial_sequences[key] = sequence
        trial_targets[key] = target
    (seqs_per_trial, _, dims) = trial_sequences[train_inds[0]].shape
    # fill in numpy arrays from dictionaries
    X_train = np.zeros((seqs_per_trial * len(train_inds), timesteps, dims))
    y_train = np.zeros(seqs_per_trial * len(train_inds))
    X_test = np.zeros((seqs_per_trial * len(test_inds), timesteps, dims))
    y_test = np.zeros(seqs_per_trial * len(test_inds))
    X_val = np.zeros((seqs_per_trial * len(val_inds), timesteps, dims))
    y_val = np.zeros((seqs_per_trial * len(val_inds)))
    i = 0
    for ind in train_inds:
        X_train[i: i + seqs_per_trial] = trial_sequences[ind]
        y_train[i: i + seqs_per_trial] = trial_targets[ind]
        i += seqs_per_trial
    i = 0
    for ind in test_inds:
        X_test[i: i + seqs_per_trial] = trial_sequences[ind]
        y_test[i: i + seqs_per_trial] = trial_targets[ind]
        i += seqs_per_trial
    i = 0
    for ind in val_inds:
        X_val[i: i + seqs_per_trial] = trial_sequences[ind]
        y_val[i: i + seqs_per_trial] = trial_targets[ind]
        i += seqs_per_trial
    return trial_sequences, trial_targets, X_train, y_train, X_test, y_test, X_val, y_val


def create_models(data, num_trials):
    """
    Creates GRU models that decode the probability of a mirror or saccade
    trial, using various lengths of LFP voltage data

    :param data: Pandas DataFrame containing LFP voltage data, number of
    trial, and mirror/saccade

    :param num_trials: number of total trials
    :return:
    """
    window_size = 700
    bin_size = 10
    dimensions = 15
    timestep_encoding = True
    steps_per_trial = 2000
    timestep_array = [100, 50, 20, 10]  # timesteps (in ms) of LFP voltage data
    # used for prediction
    center = 0
    train_inds, test_inds, val_inds = train_test_val_split(num_trials)
    for timesteps in timestep_array:
        X_trials, y_trials = create_trials(data, window_size, num_trials,
                                           steps_per_trial, timesteps,
                                           center, timestep_encoding, bin_size)
        trials, _, _ = normalize_trials(X_trials, train_inds, dimensions)
        trial_sequences, trial_targets, X_train, y_train, X_test, y_test, \
        X_val, y_val =\
        create_sequences(X_trials, y_trials, train_inds, test_inds, val_inds, timesteps)
        model = models.Sequential()
        callback = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.add(layers.CuDNNGRU(5, input_shape=(timesteps, dimensions + window_size/bin_size), return_sequences=False))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adagrad',
         loss='binary_crossentropy',
         metrics=['acc'])
        epochs = 70
        batch_size = 256
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_val, y_val),
                  callbacks=[callback])
        model.save('mirror_saccade_models/mdl_' + str(timesteps) + '.h5')


if __name__ == '__main__':
    data_filepath = "data/lfp_mirror_saccade.csv"
    data = pd.read_csv(data_filepath)
    num_trials = max(data.trial_num) + 1
    create_models(data, num_trials)