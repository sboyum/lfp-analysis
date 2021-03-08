import pandas as pd
import numpy as np
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.optimizers as optimizers


def create_sequence(data, timesteps, lag=1, pad=False, encoded_eye_direc=False, num_directions=12, start_index=0):
    """
    Creates an array of sequences and corresponding targets from a single
    trial; the sequences and targets can then be passed into a Keras RNN (
    e.g. LSTM, GRU)

    :param data: LFP voltage readings and (if encoded_eye_direc = True) eye
    direction encodings for a single trial. the first 16 columns represent LFP
    voltage readings, and the last 12 columns represent one-hot encodings of
    saccade direction. Note that if encoded_eye_direc = False, this matrix
    will only have 16 columns

    :param timesteps: number of timesteps (in ms) of LFP voltage readings
    used for prediction

    :param lag: number of timesteps (in the future, in ms) for which lfp
    predictions are generated

    :param pad: boolean indicating whether initial sequences are padded

    :param encoded_eye_direc: boolean indicating whether eye direction is
    encoded

    :param num_directions: number of saccade directions

    :param start_index: time point in trial to start using LFP data - e.g. if
    start_index = 10, predictions would not be made using the first 10 voltage
    readings in a trial. This is useful for comparing models with different
    numbers of timesteps

    :return:
    sequences: 3d numpy array, each entry represents an LFP voltage sequence
    used for prediction.

    targets: 2d numpy array, each entry represents a target used for
    prediction.
    """
    sequences = []
    targets = []
    if pad is False:
        if encoded_eye_direc is False:
            targets = data[start_index + timesteps + lag - 1:]
        else:
            targets = data[start_index + timesteps + lag - 1:, : - num_directions]
        for i in range(start_index, data.shape[0] - timesteps - lag + 1):
            a = data[i: (i + timesteps), :]
            sequences.append(a)
    else:
        if encoded_eye_direc is False:
            targets = data[lag:]
        else:
            targets = data[lag:, : - num_directions]
        for i in range(0, data.shape[0] - lag):
            a = data[max((i-timesteps+1), 0):i + 1, :]
            if i + 1 < timesteps:
                # pad sequences
                k = np.zeros((timesteps - i - 1, data.shape[1]))
                a = np.concatenate((k, a))
            sequences.append(a)
    return np.array(sequences), np.array(targets)


def create_sequences_train_test_val(trials, timesteps, train_inds, test_inds, 
                                    val_inds, keymapping, num_directions,
                                    encoded_eye_direc=True,
                                    start_index=0, lag=1,
                                    pad=False):
    """
    Creates train, test, and validation sequences for entire dataset.

    :param trials: dictionary, in which entry (i, j) is the LFP voltage
    reading (and possible saccade direction encoding) for trial j,
    and saccade direction j

    :param timesteps: number of timesteps (in ms) of LFP voltage readings
    used for prediction

    :param train_inds: list of trial indeces used for training

    :param test_inds: list of trial indeces used for testing

    :param val_inds: list of trial indeces used for validation

    :param keymapping: list that stores dictionary keys so that eye
    direction and trial number can be recovered

    :param encoded_eye_direc: boolean indicating whether eye direction is
    encoded

    :param num_directions: number of saccade directions

    :param start_index:  time point in trial to start using LFP data - e.g. if
    start_index = 10, predictions would not be made using the first 10 voltage
    readings in a trial. This is useful for comparing models with different
    numbers of timesteps

    :param lag: number of timesteps (in the future, in ms) for which lfp
    predictions are generated

    :param pad: boolean indicating whether initial sequences are padded

    :return:
    trial_sequences, trial_targets: dictionaries, where each entry is a
    sequence and target for a single trial. Dictionary entries are in the
    same format as the input variable trials above

    X_train, X_test, X_val, y_train, y_test, y_val: training, testing,
    and validation datasets
    """
    trial_sequences = dict()
    trial_targets = dict()
    for key in trials.keys():
        # create a single sequence and target from trial
        sequence, target = create_sequence(trials[key], timesteps, lag, pad,
                                           encoded_eye_direc, num_directions,
                                           start_index)
        # add sequence and target to dictionary
        trial_sequences[key] = sequence
        trial_targets[key] = target
    (seqs_per_trial, _, dims) = trial_sequences[keymapping[train_inds[0]]].shape
    # initialize train, test, and validation arrays
    X_train = np.zeros((seqs_per_trial * len(train_inds), timesteps, dims))
    y_train = np.zeros((seqs_per_trial * len(train_inds), dims - num_directions))
    X_test = np.zeros((seqs_per_trial * len(test_inds), timesteps, dims))
    y_test = np.zeros((seqs_per_trial * len(test_inds), dims - num_directions))
    X_val = np.zeros((seqs_per_trial * len(val_inds), timesteps, dims))
    y_val = np.zeros((seqs_per_trial * len(val_inds), dims - num_directions))
    i = 0
    # fill in train array
    for ind in train_inds:
        X_train[i: i + seqs_per_trial] = trial_sequences[keymapping[ind]]
        y_train[i: i + seqs_per_trial] = trial_targets[keymapping[ind]]
        i += seqs_per_trial
    i = 0
    # fill in test array
    for ind in test_inds:
        X_test[i: i + seqs_per_trial] = trial_sequences[keymapping[ind]]
        y_test[i: i + seqs_per_trial] = trial_targets[keymapping[ind]]
        i += seqs_per_trial
    i = 0
    # fill in validation array
    for ind in val_inds:
        X_val[i: i + seqs_per_trial] = trial_sequences[keymapping[ind]]
        y_val[i: i + seqs_per_trial] = trial_targets[keymapping[ind]]
        i += seqs_per_trial
    return trial_sequences, trial_targets, X_train, y_train, X_test, y_test, X_val, y_val


def normalize(trials, train_inds, keymapping, num_directions,
              encoded_eye_direc=True):
    """
    Normalizes all LFP voltages wrt to training data so that the LFP
    voltages have a mean of 0 and std of 1.

    :param trials: dictionary, in which entry (i, j) is the LFP voltage
    reading (and possible saccade direction encoding) for trial j,
    and saccade direction j

    :param train_inds: trial indeces used for training

    :param keymapping: list that stores dictionary keys so that eye
    direction and trial number can be recovered

    :param num_directions: number of saccade directions

    :param encoded_eye_direc: boolean indicating whether saccade direction
    is encoded

    :return: trials: dictionary of normalized LFP voltage data

    means, stds: channel-wise mean and standard devation of LFP voltage

    """
    i = 0
    # gather all training data into array
    for ind in train_inds:
        if i == 0:
            train_data = trials[keymapping[ind]]
        else:
            train_data = np.concatenate((train_data, trials[keymapping[ind]]))
        i += 1
    means = np.mean(train_data, axis=0)
    stds = np.std(train_data, axis=0)
    if encoded_eye_direc is False:
        num_directions = 0
    # normalize each trial
    for key in trials:
        trials[key][:, : - num_directions] = (trials[key][:, : - num_directions] - 
                                              means[: - num_directions])/stds[:- num_directions]
    return trials, means, stds

def create_trials_mapping(data, num_directions):
    """
    Creates dictionary of trials, in which each dictionary key is a tuple
    representing the eye direction and trial number

    :param data: Pandas DataFrame, contains data from all trials

    :param num_directions: number of saccade directions

    :return:
    trials: dictionary of trials

    keymapping: list of all dictionary keys, ordered by saccade direction,
    and then trial number. This is useful for splitting the dataset into
    training and testing sets.
    """
    trials = dict()
    keymapping = []
    for i in range(num_directions):
        eye_direc_subset = data[data.eye_direction == i]
        max_trial = max(eye_direc_subset.trial_num) + 1
        for j in range(max_trial):
            trials[(i, j)] = eye_direc_subset[eye_direc_subset.trial_num == j]
            trials[(i, j)] = np.array(trials[(i, j)].drop(columns=[
                'eye_direction', 'trial_num']))
            # add dictionary key to array
            keymapping.append((i, j))
    return trials, keymapping


def create_train_test_val_inds(num_trials):
    """
    Separates trial indices into train, test, and validation indeces

    :param num_trials: number of total trials, across all eye directions

    :return: train_inds, test_inds, val_inds -- lists containing train,
    test, and validation indeces, respectively
    """
    # set seed so that results are consistent
    np.random.seed(123)
    inds = np.arange(num_trials)
    np.random.shuffle(inds)
    split = .2
    test_cutoff = int(num_trials * (1 - split))
    train_inds = inds[:test_cutoff]
    test_inds = inds[test_cutoff:]
    val_cutoff = int(num_trials * (1 - split))
    val_inds = train_inds[val_cutoff:]
    train_inds = train_inds[:val_cutoff]
    return train_inds, test_inds, val_inds

def create_models(trials, train_inds, test_inds, val_inds, keymapping,
                  num_directions, encoded_eye_direc=True,
                  align_starting_timestep=True):
    """
    Creates GRU models for all combinations of lags in the array lags,
    and timesteps in timestep_array. Saves these models in .h5 format.

    :param trials: dictionary, in which entry (i, j) is the LFP voltage
    reading (and possible saccade direction encoding) for trial j,
    and saccade direction j

    :param train_inds: indeces for training trials
    :param test_inds: indeces for training trials
    :param val_inds: indeces for training trials

    :param keymapping: list of dictionary keys in trials

    :param num_directions: number of saccade directions

    :param align_starting_timestep: boolean, indicates whether we should
    adjust the start index of LFP voltages used for prediction so that
    models with different numbers of timesteps can be properly compared

    :return: None
    """
    lags = [1, 5, 10, 25, 50, 100]
    # number of GRU units used for each lag
    units = [200, 100, 75, 50, 20, 10]
    batch_size = 512
    epochs = 100
    learn_rate = .001
    optimizer = optimizers.Adam(lr=learn_rate)
    # timesteps to train models for
    timestep_array = [150, 100, 50, 20, 10, 5]
    # whether to pad early sequences in trial
    pad = False
    weights_prefix = 'lfp_models/mdl_weights'
    model_prefix = 'lfp_models/mdl'
    for timesteps in timestep_array:
        for (lag, unit) in zip(lags, units):
            timesteps_lag_pad = str(timesteps) + "_" + str(lag) + "_" + str(int(pad))
            if align_starting_timestep is True:
                start_index = max(timestep_array) - timesteps
            else:
                start_index = 0
            trial_sequences, trial_targets, X_train, y_train, X_test, y_test, X_val, y_val = \
            create_sequences_train_test_val(trials, timesteps, train_inds, test_inds, val_inds, keymapping, num_directions,
                                            encoded_eye_direc, start_index,
                                            lag, pad)
            input_dim = X_train.shape[2]
            output_dim = X_train.shape[2] - num_directions
            # saves best model - in terms of validation accuracy - from 100
            # epochs
            checkpoint = callbacks.ModelCheckpoint(filepath=weights_prefix +
                                                            timesteps_lag_pad +'.hdf5',
                                                   save_best_only=True)
            model = models.Sequential()
            model.add(layers.CuDNNGRU(units=unit, input_shape=(timesteps,
                                                               input_dim)))
            model.add(layers.Dense(output_dim))
            model.compile(optimizer=optimizer, loss='mse')
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                      validation_data=(X_val, y_val), callbacks=[checkpoint])
            # loads best model out of 100 epochs
            model.load_weights(filepath=weights_prefix +
                                        timesteps_lag_pad + '.hdf5')
            model.save(filepath=model_prefix + timesteps_lag_pad + '.h5')


if __name__ == '__main__':
    data_filepath = "data/lfp.csv"
    # boolean, indicates whether to one-hot encode eye direction
    encoded_eye_direc = True
    align_starting_timestep = True
    data = pd.read_csv(data_filepath)
    if encoded_eye_direc is True:
        data = pd.concat([data, pd.get_dummies(data.eye_direction)], axis=1)
    num_directions = 12
    num_channels = 16
    trials, keymapping = create_trials_mapping(data, num_directions)
    num_trials = len(keymapping)
    # gets training and testing indeces
    train_inds, test_inds, val_inds = create_train_test_val_inds(num_trials)
    # normalizes trials
    trials, means, stds = normalize(trials, train_inds, keymapping,
                                    num_directions, encoded_eye_direc)
    create_models(trials, train_inds, test_inds, val_inds, keymapping,
                  num_directions, encoded_eye_direc, align_starting_timestep)
