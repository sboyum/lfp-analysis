from sklearn import metrics
import tensorflow.keras.models as models
from train_spikes import *

def error_analysis(trials, train_inds, test_inds, val_inds, num_dimensions, num_neurons):
    """
    Performs error analysis for spike prediction to save time in the future
    -- calculates auc's and accuracies by timesteps, lag, bin size, and time point
    in trial

    :param trials:  LFP voltage and spike history, in dictionary format,
    where each entry represents data for a particular trial

    :param train_inds: list of trial indeces used for training
    :param test_inds: list of trial indeces used for testing
    :param val_inds: list of trial indeces used for validation

    :param num_dimensions: number of LFP voltage channels
    :param num_neurons: number of neurons
    :return:
    """
    timestep_array = [100, 50, 10]  # amount of time (in ms) of LFP voltage
    # data and spike history used for prediction
    lags = [1, 5, 10, 50]  # amount of time (in ms) in the future for which
    # we try to predict spikes
    bin_sizes = [1, 5, 10, 50] # length of time period (in ms) for
    # prediction of spike occurrence
    accuracies = np.zeros((len(timestep_array), len(lags), len(bin_sizes)))
    # average prediction accuracy for spike occurrences, broken up by timesteps,
    # lag, and bin size
    aucs = np.zeros((len(timestep_array), len(lags), len(bin_sizes)))  # AUC
    #  average for spike occurrences, broken up by timesteps, lag, and bin size
    auc_mins = np.zeros((len(timestep_array), len(lags), len(bin_sizes)))  #
    # the minumum AUC, across all neurons, for each combination of timestep,
    #  lag and bin size
    auc_maxs = np.zeros((len(timestep_array), len(lags), len(bin_sizes)))  #
    # the maximum AUC, across all neurons, for each combination of timestep,
    #  lag, and bin size
    for i in range(len(timestep_array)):
        for j in range(len(lags)):
            for k in range(len(bin_sizes)):
                timesteps = timestep_array[i]
                lag = lags[j]
                bin_size = bin_sizes[k]
                timesteps_lag_bin_size = str(timesteps) + '_' + str(lag) + '_' + str(bin_size)
                model = models.load_model('spike_models/mdl_' +
                                          timesteps_lag_bin_size + '.h5')
                trial_sequences, trial_targets, X_train, y_train, X_test, y_test, X_val, y_val =\
                create_sequences(trials, train_inds, test_inds, val_inds, timesteps, num_neurons, lag, num_dimensions, bin_size)
                y_pred = model.predict(X_test)
                y_pred_rounded = np.rint(y_pred)
                accuracies[i, j, k] = metrics.accuracy_score(y_test.reshape(num_neurons * len(y_test)), y_pred_rounded.reshape(num_neurons * len(y_test)))
                fpr, tpr, _ = metrics.roc_curve(y_test.reshape((num_neurons * len(y_test))), y_pred.reshape((num_neurons * len(y_test))))
                auc = metrics.auc(fpr, tpr)
                aucs[i, j, k] = auc
                auc_by_neuron = [metrics.roc_auc_score(y_test[:, i], y_pred[:, i]) for i in range(num_neurons)]
                auc_mins[i, j, k] = np.min(auc_by_neuron)
                auc_maxs[i, j, k] = np.max(auc_by_neuron)
                # save false positive and true positive rate
                np.save('spike_errors/fpr_' + timesteps_lag_bin_size + '.npy',
                        fpr)
                np.save('spike_errors/tpr_' + timesteps_lag_bin_size + '.npy',
                        tpr)
                trial_length = trial_targets[0].shape[0]
                aucs_by_time = np.zeros(trial_length)
                y_pred_time = []
                y_test_time = []
                for ind in test_inds:
                    y_test_time.append(trial_targets[ind])
                    y_pred_time.append(model.predict(trial_sequences[ind]))
                y_test_time = np.array(y_test_time)
                y_pred_time = np.array(y_pred_time)
                for n in range(trial_length):
                    auc = metrics.roc_auc_score(np.reshape(y_test_time[:,n,:], len(test_inds) * num_neurons),
                        np.reshape(y_pred_time[:,n,:], len(test_inds) * num_neurons))
                    aucs_by_time[n] = auc
                np.save('spike_errors/auc_time_' + timesteps_lag_bin_size +
                        '.npy', aucs_by_time)  # saves AUC, broken up by
                # time point in trial
                del trial_sequences, trial_targets, X_train, y_train, X_test, y_test, X_val, y_val, fpr, tpr
    np.save('spike_errors/aucs.npy', aucs)
    np.save('spike_errors/accuracies.npy', accuracies)
    np.save('spike_errors/auc_mins.npy', auc_mins)
    np.save('spike_errors/auc_maxs.npy', auc_maxs)

if __name__ == '__main__':
    data_filepath = "data/lfp_spikes.csv"
    data = pd.read_csv(data_filepath)
    num_dimensions = 15
    num_neurons = 52
    num_trials = max(data.trial_index) + 1
    trials = create_trials(data, num_trials)
    train_inds, test_inds, val_inds = train_test_val_split(num_trials)
    trials, _, _ = normalize_trials(trials, train_inds, num_dimensions)
    error_analysis(trials, train_inds, test_inds, val_inds, num_dimensions, num_neurons)