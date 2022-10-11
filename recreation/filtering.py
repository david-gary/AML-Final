from scipy import signal
from data_loader import save_pickled_data
import pandas as pd
import numpy as np

# Function for the FIR filter, hamming window of 212


def filter_data(train_data, eval_data, filter_order=5, cutoff_low=4, cutoff_high=48, sample_rate=250):
    # filter_order is the number of taps in the filter
    # cutoff_low is the low cutoff frequency (4 Hz for ES1D)
    # cutoff_high is the high cutoff frequency (48 Hz for ES1D)
    # sample_rate is the sampling rate of the data

    # prepare two lists to hold the filtered dataframes
    # - items in each list will be tuples of (df_filtered, age_label)
    train_data_filtered = []
    eval_data_filtered = []

    # b is the filter coefficients
    # a is the denominator coefficients

    # create the filter
    b = signal.firwin(
        filter_order, [cutoff_low, cutoff_high], pass_zero=False, fs=sample_rate)
    a = 1  # no denominator coefficients

    # filter each channel in the train data
    for tup in train_data:

        data, age = tup[0], tup[1]

        # create a new dataframe to hold the filtered data
        df_filtered = pd.DataFrame()

        # filter each channel
        for col in data.columns:
            df_filtered[col] = signal.filtfilt(
                b, a, data[col], padlen=0)

        train_data_filtered.append((df_filtered, age))

    # filter each channel in the eval data
    for tup in eval_data:

        data, age = tup[0], tup[1]

        # create a new dataframe to hold the filtered data
        df_filtered = pd.DataFrame()

        # filter each channel
        for col in data.columns:
            df_filtered[col] = signal.filtfilt(
                b, a, data[col], padlen=0)

        eval_data_filtered.append((df_filtered, age))

    # save the dataframes to pickle files for faster loading
    save_pickled_data(train_data_filtered, eval_data_filtered, "filtered")

    return train_data_filtered, eval_data_filtered


# Function to segment data into n second windows (default is 6)
# - performs PSD calculation on each window
# - returns two tuples of (psd, age_label), one for train data and one for eval data
def welch_data(train_data, eval_data, window_size=6, sample_rate=250):
    # window_size is the size of the window in seconds
    # sample_rate is the sampling rate of the data

    # calculate the number of samples in the window
    window_samples = window_size * sample_rate

    # prepare two lists to hold the PSD dataframes
    # - items in each list will be tuples of (df_psd, age_label)
    train_data_final = []
    eval_data_final = []

    # each item in a dataset is a tuple of (dataframe, age_label)
    # - loop through each item in the train data
    for tup in train_data:
        # create a list to hold the PSD dataframes
        # - each item in the list will be a tuple of (df_psd, age_label)
        train_data_psd = []

        # get the dataframe and age label
        data, age = tup[0], tup[1]

        # loop through each channel

        for col in data.columns:
            # create a new dataframe to hold the PSD data
            df_psd = pd.DataFrame()

            # loop through the signal segment
            for i in range(0, len(data[col]), window_samples):
                # get the signal segment
                signal_segment = data[col][i:i+window_samples]

                # calculate the PSD
                psd = calculate_psd(signal_segment)

                # add the PSD to the dataframe
                df_psd = df_psd.append(pd.Series(psd), ignore_index=True)

            # add the dataframe to the list
            train_data_psd.append((df_psd, age))

        # add the list to the final list
        train_data_final.append(train_data_psd)

    # each item in a dataset is a tuple of (dataframe, age_label)
    # - loop through each item in the eval data
    for tup in eval_data:
        # create a list to hold the PSD dataframes
        # - each item in the list will be a tuple of (df_psd, age_label)
        eval_data_psd = []

        # get the dataframe and age label
        data, age = tup[0], tup[1]

        # loop through each channel
        for col in data.columns:
            # create a new dataframe to hold the PSD data
            df_psd = pd.DataFrame()

            # loop through the signal segment
            for i in range(0, len(data[col]), window_samples):
                # get the signal segment
                signal_segment = data[col][i:i+window_samples]

                # calculate the PSD
                psd = calculate_psd(signal_segment)

                # add the PSD to the dataframe
                df_psd = df_psd.append(pd.Series(psd), ignore_index=True)

            # add the dataframe to the list
            eval_data_psd.append((df_psd, age))

        # add the list to the final list
        eval_data_final.append(eval_data_psd)

    # save the dataframes to pickle files for faster loading
    save_pickled_data(train_data_final, eval_data_final, "welch")


# Function to calculate the power spectral density of a signal segment
def calculate_psd(signal_segment, sample_rate=250):
    # signal_segment is a 1D array
    # sample_rate is the sampling rate of the data

    # calculate the power spectral density
    _, psd = signal.welch(signal_segment, sample_rate)

    return psd


# Function for FFT calculation of a signal segment written from scratch
def fft(signal_segment, sample_rate=250):
    #
    # get the number of samples
    N = len(signal_segment)

    # create a list to hold the FFT values
    fft = []

    # loop through the frequency bins
    for k in range(N):
        # calculate the sum for the kth frequency bin
        sum = 0
        for n in range(N):
            sum += signal_segment[n] * np.exp(-1j*2*np.pi*k*n/N)

        # add the sum to the list of FFT values
        fft.append(sum)

    # calculate the frequencies by dividing the sample rate by the number of samples in the signal segment
    freqs = np.linspace(0, sample_rate, N)

    return freqs, fft


# Function to calculate the power spectral density of a signal segment, written from scratch
def calculate_psd_scratch(signal_segment, sample_rate=250):
    # signal_segment is a 1D array
    # sample_rate is the sampling rate of the data

    # calculate the power spectral density using welch's method
    # - calculate the periodogram of the signal segment
    # - average the periodograms of the signal segment

    # calculate the periodogram of the signal segment using the FFT
    # - FFT is also written from scratch
    periodogram = fft(signal_segment)

    # calculate the number of unique points
    num_unique_pts = int(np.ceil((len(signal_segment)+1)/2.0))

    # take the first half of the periodogram
    periodogram = periodogram[:num_unique_pts]

    # multiply the first half by two (since we dropped the second half)
    periodogram *= 2

    # account for the fact that we dropped the DC component
    if len(signal_segment) % 2 == 0:
        periodogram[-1] /= 2

    # calculate the frequencies
    freqs = np.arange(0, num_unique_pts, 1.0) * \
        (sample_rate / len(signal_segment))

    # calculate the power spectral density
    psd = periodogram / sample_rate

    return freqs, psd
