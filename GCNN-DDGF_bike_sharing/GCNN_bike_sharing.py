import numpy as np
import pandas as pd
import pickle
import datetime
import sys
import os

from utils import normalize_adj, StandardScaler
from gcn import gcn, gcnn_ddgf


def main(demand_type):
    file_name = '../../bikeshare-experiments/data/bikeshare/%s_demand.pickle'.format(demand_type)
    # Import Data
    fileObject = open(file_name, 'rb')
    hourly_bike = pickle.load(fileObject)
    hourly_bike = pd.DataFrame(hourly_bike)


    # Split Data into Training, Validation and Testing
    node_num = hourly_bike.shape[1] # node number
    feature_in = 24 # number of features at each node, e.g., bike sharing demand from past 24 hours
    horizon = 24 # the length to predict, e.g., predict the future one hour bike sharing demand

    X_whole = []
    Y_whole = []

    x_offsets = np.sort(
        np.concatenate((np.arange(-feature_in + 1, 1, 1),))
    )

    y_offsets = np.sort(np.arange(1, 1 + horizon, 1))

    min_t = abs(min(x_offsets))
    max_t = abs(hourly_bike.shape[0] - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = hourly_bike.iloc[t + x_offsets, 0:node_num].values.flatten('F')
        y_t = hourly_bike.iloc[t + y_offsets, 0:node_num].values.flatten('F')
        X_whole.append(x_t)
        Y_whole.append(y_t)

    X_whole = np.stack(X_whole, axis=0)
    Y_whole = np.stack(Y_whole, axis=0)
    X_whole = np.reshape(X_whole, [X_whole.shape[0], node_num, feature_in])

    # split the dataset into train val test
    num_samples = X_whole.shape[0]
    num_train = num_samples - 60 * 24
    num_val = 30 * 24
    num_test = 30 * 24

    # Train
    X_training = X_whole[:num_train, :]
    Y_training = Y_whole[:num_train, :]

    # shuffle the training dataset
    perm = np.arange(X_training.shape[0])
    np.random.shuffle(perm)
    X_training = X_training[perm]
    Y_training = Y_training[perm]

    # Validation
    X_val = X_whole[num_train:num_train+num_val:horizon, :]
    Y_val = Y_whole[num_train:num_train+num_val:horizon, :]

    # Test
    X_test = X_whole[num_train+num_val:num_train+num_val+num_test:horizon, :]
    Y_test = Y_whole[num_train+num_val:num_train+num_val+num_test:horizon, :]


    scaler = StandardScaler(mean=X_training.mean(), std=X_training.std())

    X_training = scaler.transform(X_training)
    Y_training = scaler.transform(Y_training)

    X_val = scaler.transform(X_val)
    Y_val = scaler.transform(Y_val)

    X_test = scaler.transform(X_test)
    Y_test = scaler.transform(Y_test)

    # Hyperparameters
    learning_rate = 0.01 # learning rate
    decay = 0.9
    batchsize = 100 # batch size

    hidden_num_layer = [10, 10, 20] # determine the number of hidden layers and the vector length at each node of each hidden layer
    reg_weight = [0, 0, 0] # regularization weights for adjacency matrices L1 loss

    keep = 1 # drop out probability

    early_stop_th = 200 # early stopping threshold, if validation RMSE not dropping in continuous 20 steps, break
    training_epochs = 500 # total training epochs

    # Training
    start_time = datetime.datetime.now()

    val_error, predic_val, test_Y, test_error, bestWeightA = gcnn_ddgf(hidden_num_layer, reg_weight, node_num, feature_in, horizon, learning_rate, decay, batchsize, keep, early_stop_th, training_epochs, X_training, Y_training, X_val, Y_val, X_test, Y_test, scaler, 'RMSE')

    end_time = datetime.datetime.now()

    print('Total training time: ', end_time-start_time)

    # Extract results
    if os.path.exists("../../bikeshare-experiments/results/GCNN"):
        os.makedirs("../../bikeshare-experiments/results/GCNN")

    # Compute Validation RMSE by day.
    rmse_frame = pd.DataFrame(columns=['time_step', 'station_index', 'RMSE'])
    for time_step in np.arange(30):
        for station_index in np.arange(node_num):
            # Compute station RMSE
            rmse = np.sqrt(np.sum(
                np.square(predic_val[time_step, station_index * 24: station_index * 24 + 24]
                          - scaler.inverse_transform(Y_val)[time_step, station_index * 24: station_index * 24 + 24]) / 24))
            rmse_series = pd.Series({
                'time_step': time_step,
                'station_index': station_index,
                'RMSE': rmse
            })
            rmse_frame = rmse_frame.append(rmse_series, ignore_index=True)
    rmse_frame.to_csv("../../bikeshare-experiments/results/GCNN/validation_daily_station_rmse.csv")

    # Compute Test RMSE by day.
    rmse_frame = pd.DataFrame(columns=['time_step', 'station_index', 'RMSE'])
    for time_step in np.arange(30):
        for station_index in np.arange(node_num):
            # Compute station RMSE
            rmse = np.sqrt(np.sum(
                np.square(test_Y[time_step, station_index * 24: station_index * 24 + 24]
                          - scaler.inverse_transform(Y_test)[time_step, station_index * 24: station_index * 24 + 24]) / 24))
            rmse_series = pd.Series({
                'time_step': time_step,
                'station_index': station_index,
                'RMSE': rmse
            })
            rmse_frame = rmse_frame.append(rmse_series, ignore_index=True)
    rmse_frame.to_csv("../../bikeshare-experiments/results/GCNN/test_daily_station_rmse.csv")

    np.savetxt("../../bikeshare-experiments/results/GCNN/%s_prediction_validation.csv".format(demand_type), predic_val, delimiter = ',')
    np.savetxt("../../bikeshare-experiments/results/GCNN/%s_prediction_test.csv".format(demand_type), test_Y, delimiter = ',')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python2 GCNN_bike_sharing.py <arrival or departure>")

    main(sys.argv[1])
