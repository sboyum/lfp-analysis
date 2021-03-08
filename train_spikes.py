import pandas as pd
import numpy as np
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.callbacks as callbacks

def create_sequence(data, timesteps, lag, num_dimensions, num_neurons, bin_size):
    """
    Creates an array of sequences and corresponding targets for a single
    trials, formatted so that they can be passed into a Keras RNN

    :param data: NumPy array of LFP Voltage Information concatenated with
    spike
    history.
    Total dimension size is 67 (15 for LFP voltage, 52 for spike history).
    Each spike history entry is binary, indicating whether a spike occured
    in a particular neuron at a given timestep

    :param timesteps: number of ms to look back for prediction (LFP voltage
    and spike history)

    :param lag: number of timesteps (in the future, in ms) for which spike
    predictions are generated

    :param num_dimensions: number of electrode channels

    :param num_neurons: number of neurons for which spike history is recorded

    :param bin_size: length of time period for spike prediction

    :return:
    sequences: 3d numpy array, LFP voltage + spike history used for prediction

    targets: 1d array (containing num_neurons elements), each entry represents
    spike information associated with lag and bin size
    """
    # create sequences
    sequences = []
    for i in range(data.shape[0] - timesteps - lag - bin_size + 2):
        a = data[i:(i + timesteps)]
        sequences.append(a)
    init_targets = np.array(data[timesteps + lag - 1:, num_dimensions:])
    # create targets
    targets = np.zeros((len(sequences), num_neurons))
    for i in range(len(sequences)):
        targets[i,:] = np.max(init_targets[i:i + bin_size], axis=0)
    return np.array(sequences), targets

def create_trials(data, num_trials):
    """
    :param data: LFP Voltage Information concatenated with spike history.
    Total dimension size is 67 (15 for LFP voltage, 52 for spike history).
    Each spike history entry is binary, indicating whether a spike occured
    in a particular neuron at a given timestep

    :param num_trials: number of trials in dataset

    :return: the input data in dictionary format, where each entry
    represents LFP voltage data and spike history for a trial, and each
    trial has an associated index
    """
    trials = dict()
    for i in range(num_trials):
        trials[i] = data[data.trial_index == i]
        trials[i] = np.array(trials[i].drop(columns=['trial_index']))
    return trials

def train_test_val_split(num_trials):
    """
    Splits trials into train, test, and validation trials

    :param num_trials: number of trials in dataset

    :return: three separate lists, indicating the trial indeces of train,
    test, and validation trials
    """
    inds = np.arange(num_trials)
    # set seed so that results are consistent
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

    :param trials: LFP voltage and spike history, in dictionary format,
    where each entry represents data for a particular trial

    :param train_inds: trial indeces used for training the model

    :param dimensions: number of channels for LFP

    :return:

    trials: Normalized LFP voltage and spike history in dictionary format

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
        trials[key][:,:dimensions] = (trials[key][:, :dimensions,] - means[:dimensions])/stds[:dimensions]
    return trials, means, stds

def create_sequences(trials, train_inds, test_inds, val_inds, timesteps, num_neurons, lag, num_dimensions,
                    bin_size):
    """
    Creates train, test, and validation sequences for entire dataset.

    :param trials:  LFP voltage and spike history, in dictionary format,
    where each entry represents data for a particular trial

    :param train_inds: list of trial indeces used for training

    :param test_inds: list of trial indeces used for testing

    :param val_inds: list of trial indeces used for validation

    :param timesteps: number of timesteps (in ms) of LFP voltage readings
    and spike history used for prediction

    :param num_neurons: number of neurons for which spikes are recorded

    :param lag: number of timesteps (in the future, in ms) for which spike
    predictions are generated

    :param num_dimensions: number of LFP voltage channels

    :param bin_size: time period (in ms) for which we predict the
    probability of a spike occurring

    :return: trial_sequences, trial_targets: dictionaries, where each entry
    is a sequence and target for a single trial

    X_train, X_test, X_val, y_train, y_test, y_val: train, test and
    validation data that can be directly passed into a Keras RNN to train it.
    """
    trial_sequences = dict()
    trial_targets = dict()
    # create sequences and targets and put them in dictionaries
    for key in trials.keys():
        sequence, target = create_sequence(trials[key], timesteps, lag, num_dimensions, num_neurons, bin_size)
        trial_sequences[key] = sequence
        trial_targets[key] = target
    (seqs_per_trial, _, dims) = trial_sequences[train_inds[0]].shape
    # iterate through dictionaries and load them into arrays
    X_train = np.zeros((seqs_per_trial * len(train_inds), timesteps, dims))
    y_train = np.zeros((seqs_per_trial * len(train_inds), num_neurons))
    X_test = np.zeros((seqs_per_trial * len(test_inds), timesteps, dims))
    y_test = np.zeros((seqs_per_trial * len(test_inds), num_neurons))
    X_val = np.zeros((seqs_per_trial * len(val_inds), timesteps, dims))
    y_val = np.zeros((seqs_per_trial * len(val_inds), num_neurons))
    i = 0
    # fill in train array
    for ind in train_inds:
        X_train[i: i + seqs_per_trial] = trial_sequences[ind]
        y_train[i: i + seqs_per_trial] = trial_targets[ind]
        i += seqs_per_trial
    i = 0
    # fill in test array
    for ind in test_inds:
        X_test[i: i + seqs_per_trial] = trial_sequences[ind]
        y_test[i: i + seqs_per_trial] = trial_targets[ind]
        i += seqs_per_trial
    i = 0
    # fill in validation arrays
    for ind in val_inds:
        X_val[i: i + seqs_per_trial] = trial_sequences[ind]
        y_val[i: i + seqs_per_trial] = trial_targets[ind]
        i += seqs_per_trial
    return trial_sequences, trial_targets, X_train, y_train, X_test, y_test, X_val, y_val

def create_models(trials, train_inds, test_inds, val_inds, num_dimensions, num_neurons):
    """
    Trains multiple LSTM models for spike prediction for all combinations in timestep_array,
    lag, and bin_sizes

    :param trials:LFP voltage and spike history, in dictionary format,
    where each entry represents data for a particular trial

    :param train_inds: list of trial indeces used for training

    :param test_inds: list of trial indeces used for testing

    :param val_inds: list of trial indeces used for validation

    :param num_dimensions: number of LFP voltage channels

    :param num_neurons: number of neurons for spike prediction
    :return:
    """
    timestep_array = [100, 50, 10]
    lags = [1, 5, 10, 50]
    bin_sizes = [1, 5, 10, 50]
    for timesteps in timestep_array:
        for lag in lags:
            for bin_size in bin_sizes:
                timesteps_lag_bin_size = str(timesteps) + '_' + str(lag) + '_' + str(bin_size)
                trial_sequences, trial_targets, X_train, y_train, X_test, y_test, X_val, y_val =\
                create_sequences(trials, train_inds, test_inds, val_inds, timesteps, num_neurons, lag, num_dimensions,
                    bin_size)
                model = models.Sequential()
                model.add(layers.CuDNNLSTM(50))
                model.add(layers.Dense(num_neurons, activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
                epochs = 20
                batch_size = 256
                callback = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[callback])
                model.save('spike_models/mdl_' + timesteps_lag_bin_size +
                           '.h5')
                del trial_sequences, trial_targets, X_train, y_train, X_test, y_test, X_val, y_val

if __name__ == '__main__':
    data_filepath = "data/lfp_spikes.csv"
    data = pd.read_csv(data_filepath)
    num_dimensions = 15
    num_neurons = 52
    num_trials = max(data.trial_index) + 1
    trials = create_trials(data, num_trials)
    # splits into training, testing, and validation indeces
    train_inds, test_inds, val_inds = train_test_val_split(num_trials)
    # normalizes trials
    trials, _, _ = normalize_trials(trials, train_inds, num_dimensions)
    create_models(trials, train_inds, test_inds, val_inds, num_dimensions, num_neurons)
