from train_lfp import *
import tensorflow.keras.models as models
from sklearn import metrics


def get_sequences_target(seqs, targets, indeces, keymapping):
    """
    Returns sequences of LFP voltage data and associated targets (i.e.,
    what would be used for prediction in a Keras RNN model) for given
    trial indeces

    :param seqs: dictionary, in which each entry is an array of prediction
    sequences for a given trial

    :param targets: dictionary, in which each entry is an array of
    prediction targets for a given trial

    :param indeces: list of trial indeces to obtain sequences from

    :param keymapping: list of all dictionary keys, ordered by saccade
    direction, and then trial number

    :return: out_seq, out_target: NumPy arrays containing sequences and
    targets (that could be used to train a Keras RNN) for the given indeces

    """
    sequence_length = seqs[keymapping[0]].shape[0]
    num_dimensions = targets[keymapping[0]].shape[1]
    timesteps = seqs[keymapping[0]].shape[1]
    in_dims = seqs[keymapping[0]].shape[2]
    out_dims = targets[keymapping[0]].shape[1]
    out_seq = np.zeros((sequence_length * len(indeces), timesteps, in_dims))
    out_target = np.zeros((sequence_length * len(indeces), out_dims))
    for i in range(len(indeces)):
        ind = indeces[i]
        start = i * sequence_length
        end = (i + 1) * sequence_length
        out_seq[start : end] = seqs[keymapping[ind]]
        out_target[start : end] = targets[keymapping[ind]]
    return out_seq, out_target


def error_analysis(trials, train_inds, test_inds, val_inds, keymapping, num_directions):
    """
    This function assumes that models have been built for all
    combinations of lags, units, and timesteps in the associated arrays
    below. Creates and saves several numpy arrays that calculate mean square
    by number of timesteps, eye direction, channel etc. to save computation
    time later

    :param trials: dictionary, in which entry (i, j) is the LFP voltage
    reading (and possible saccade direction encoding) for trial j,
    and saccade direction j

    :param train_inds: indeces for training trials
    :param test_inds: indeces for training trials
    :param val_inds: indeces for training trials

    :param keymapping: list of dictionary keys in trials

    :param num_directions: number of saccade directions
    :return:
    """
    lags = [1, 5, 10, 25, 50, 100] # prediction lags
    units = [200, 100, 75, 50, 20, 10] # number of GRU units
    timestep_array = [150, 100, 50, 20, 10, 5] # number of timesteps (in ms)
    #  used for GRU models
    align_starting_timestep = True  # indicates whether we adjust the start
    # point of the trial for different timesteps (more explanation in paper)
    pad = False
    # full_errors: breaks up the errors by timestep, lag, eye direction,
    # and channel. So, full_errors[i][j][k][l] would be the MSE for the ith
    # entry in the array "timestep_array", the jth entry in the array
    # "lags", the kth eye direction, and the lth channel
    full_errors = []
    # avg_errors: averages the errors by eye direction and channel; errors
    # are broken up by timesteps and lag. So avg_errors[i][j] would be the
    # average MSE for the ith entry in "timestep_array" and the jth entry in
    #  "lags"
    avg_errors = []
    # errors_by_direction: averages the error by channel; errors are broken
    #  up by timesteps, lag, and saccade direction. So
    # errors_by_direction[i][j][k] would be the average MSE for the ith
    # entry in "timestep_array", the jth entry in "lags", and the kth eye
    # direction
    errors_by_direction = []
    # errors_by_channel: averages the error by saccade direction; errors are
    # broken up by timesteps, lag, and channel. So errors_by_direction[i][j][k]
    # would be the average MSE for the ith entry in "timestep_array", the jth
    # entry in "lags", and the kth channel
    errors_by_channel = []
    # errors_by_channel_time - same as error_by_channel, but errors are also
    #  broken up by time point in the trial
    errors_by_channel_time = []
    # errors_by_direction_time - same as error_by_direction, but errors are
    # also broken up by time point in the trial
    errors_by_direction_time = []
    # errors_by_time - errors broken up by time, but averaged over eye
    # direction and channel. So errors_by_direction_time[i][j][k] would be
    # the MSE for the ith entry in "timestep_array", the jth entry in
    # "lags", and the kth time point in the trial
    errors_by_time = []
    model_file_prefix = "lfp_models/mdl"  # folder from which to load
    # files
    for timesteps in timestep_array:
        errors = []
        avg_error = []
        error_by_channel = []
        error_by_direction = []
        error_by_time = []
        error_by_channel_time = []
        error_by_direction_time = []
        j = 0
        if align_starting_timestep is True:
            start_index = max(timestep_array) - timesteps
        else:
            start_index = 0
        for (lag, unit) in zip(lags, units):
            print(timesteps, lag)
            timesteps_lag_pad = str(timesteps) + "_" + str(lag) + "_" + str(int(pad))
            model = models.load_model(model_file_prefix + timesteps_lag_pad + '.h5')
            trial_sequences, trial_targets, X_train, y_train, X_test, y_test, X_val, y_val = \
            create_sequences_train_test_val(trials, timesteps, train_inds,
                                            test_inds, val_inds, keymapping,
                                            num_directions, lag, pad, start_index)
            sequence_length = trial_sequences[keymapping[0]].shape[0]
            num_channels = y_train.shape[1]
            y_pred = model.predict(X_test)
            avg_error.append(metrics.mean_squared_error(y_test, y_pred))
            error_by_channel.append(np.mean(np.square(y_test - y_pred), axis = 0))
            error_array = np.zeros((num_directions, num_channels))
            error_by_time.append(np.zeros((sequence_length)))
            error_by_channel_time.append(np.zeros((num_channels, sequence_length)))
            error_by_direction_time.append(np.zeros((num_directions, sequence_length)))
            num_instances_direcs = np.zeros((num_directions))
            for i in range(len(test_inds)):
                ind = test_inds[i]
                sequence, target = get_sequences_target(trial_sequences, trial_targets, [ind], keymapping)
                prediction = model.predict(sequence)
                error_by_time[j] += np.mean(np.square(target - prediction), axis = 1)
                error_by_channel_time[j] += np.square(target - prediction).T
                direc = keymapping[ind][0]
                error_by_direction_time[j][direc] += np.mean(np.square(target - prediction), axis = 1)
                num_instances_direcs[direc] += 1
            error_by_direction_time[j] /= num_instances_direcs[:, np.newaxis]
            error_by_time[j] /= len(test_inds)
            error_by_channel_time[j] /= len(test_inds)

            direction_error = []
            for i in range(num_directions):
                test_inds_direc = [ind for ind in test_inds if keymapping[ind][0] == i]
                X_direc, y_test_direc = get_sequences_target(trial_sequences, trial_targets,
                                                                           test_inds_direc, keymapping)
                y_pred_direc = model.predict(X_direc)
                direction_error.append(metrics.mean_squared_error(y_test_direc, y_pred_direc))
                for k in range(num_channels):
                    error_array[i, k] = metrics.mean_squared_error(
                        y_test_direc[:, k], y_pred_direc[:, k])
            errors.append(error_array)
            error_by_direction.append(direction_error)
            j += 1
        full_errors.append(errors)
        avg_errors.append(avg_error)
        errors_by_channel.append(error_by_channel)
        errors_by_direction.append(error_by_direction)
        errors_by_time.append(error_by_time)
        errors_by_channel_time.append(error_by_channel_time)
        errors_by_direction_time.append(error_by_direction_time)

    np.save("lfp_errors/full_errors.npy", full_errors)
    np.save("lfp_errors/avg_errors.npy", avg_errors)
    np.save("lfp_errors/errors_by_channel.npy", errors_by_channel)
    np.save("lfp_errors/errors_by_direction.npy",
            errors_by_direction)
    np.save("lfp_errors/errors_by_time.npy", errors_by_time)
    for i in range(len(timestep_array)):
        for j in range(len(lags)):
            timestep_lag = str(timestep_array[i]) + "_" + str(lags[j])
            np.save("lfp_errors/errors_by_channel_time" +
                    timestep_lag + ".npy", errors_by_channel_time[i][j])
            np.save("lfp_errors/errors_by_direction_time" +
                    timestep_lag + ".npy", errors_by_direction_time[i][j])


if __name__ == '__main__':
    data_filepath = "data/lfp.csv"  # filepath for .csv file to load
    data = pd.read_csv(data_filepath)
    data = pd.concat([data, pd.get_dummies(data.eye_direction)], axis=1)
    num_directions = 12
    num_channels = 16
    trials, keymapping = create_trials_mapping(data, num_directions)
    num_trials = len(keymapping)
    train_inds, test_inds, val_inds = create_train_test_val_inds(num_trials)
    trials, means, stds = normalize(trials, train_inds, keymapping,
                                    num_directions, num_channels)
    error_analysis(trials, train_inds, test_inds, val_inds, keymapping, num_directions)