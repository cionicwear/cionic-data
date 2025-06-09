"""
Processing and filtering functions for EMG and Impedance data

Contents:
    + MODULES: FFT
    + MODULES: RMS
    + FILTERS
    + PROCESS DATA
    + DATA CHECKS
"""

import numpy as np
import pandas as pd
from scipy import integrate, signal

###############################################################################
#                                                                             #
#                       FREQUENCY DOMAIN FEATURES                             #
#                                                                             #
###############################################################################


def calculate_welch_density(time_data, sampling_freq, nperseg_value=1024):
    """
    Calculate Welch Power Spectrum Density (PSD)
        Units: V^2/Hz
        Inputs: pandas Series or numpy arrays
        Outputs: numpy arrays
    """
    freq, data_welch_PSD = signal.welch(
        time_data,
        sampling_freq,
        nperseg=nperseg_value,
        scaling="density",
        nfft=nperseg_value * 10,
    )
    return freq, data_welch_PSD


def calculate_welch_spectrum(time_data, sampling_freq, nperseg_value=1024):
    """
    Calculate Welch Power Spectrum Density (PSD) on linear scale
        Units: V^2
        Inputs: pandas Series or numpy arrays
        Outputs: numpy arrays
    """
    freq, data_welch_power = signal.welch(
        time_data,
        sampling_freq,
        nperseg=nperseg_value,
        scaling="spectrum",
        nfft=nperseg_value * 10,
    )
    return freq, data_welch_power


def dB_scale(data):
    data_dB = 10 * np.log10(data)
    return data_dB


def normalize_fft(fft_data, freq):
    """
    Normalize an FFT dataset so area under the FFT = 1
        data = numpy array or pandas series
        norm_data = numpy array
    """
    trapz = integrate.trapz(fft_data, freq)
    norm_data = fft_data / trapz
    return norm_data


def calculate_cdf(fft_data, freq):
    """
    Calculate normalized CDF of Welch Density FFT
        data = numpy array or pandas series
        NOTE: len(cdf) = len(data) - 1  Make sure to account for this in plotting
        plot(freq[1::], cdf)
    """
    norm_data = normalize_fft(fft_data, freq)
    cdf = integrate.cumtrapz(norm_data, freq)
    return cdf


def median_freq(rawEMGPowerSpectrum, frequencies):
    """From 'pysiology' module (can't import module directly because matplotlib.pyplot
    has compatiblity issues with pyqtgraph)
    https://pypi.org/project/pysiology/

    Obtain the Median Frequency of the PSD (FFT Signal).

    MDF is a frequency at which the spectrum is divided into two regions with equal
    amplitude, in other words, MDF is half of TTP feature

    * Input:
        * raw EMG Power Spectrum
        * frequencies
    * Output:
        * Median Frequency  (Hz)

    :param rawEMGPowerSpectrum: power spectrum of the EMG signal
    :type rawEMGPowerSpectrum: list
    :param frequencies: frequencies of the PSD
    :type frequencies: list
    :return: median frequency of the EMG power spectrum
    :rtype: float
    """
    MDP = sum(rawEMGPowerSpectrum) * (1 / 2)
    for i in range(1, len(rawEMGPowerSpectrum)):
        if sum(rawEMGPowerSpectrum[0:i]) >= MDP:
            MDF = frequencies[i]
            return MDF


def mean_freq(rawEMGPowerSpectrum, frequencies):
    """From pysiology module (can't import module directly because matplotlib.pyplot
    has compatiblity issues with pyqtgraph)
    https://pypi.org/project/pysiology/

    Calculate mean frequency of the FFT Signal
    Obtain the mean frequency of the EMG signal, evaluated as the sum of
    product of the EMG power spectrum and the frequency divided by total sum of the
    spectrum intensity::

        MNF = sum(fPj) / sum(Pj) for j = 1 -> M
        M = length of the frequency bin
        Pj = power at freqeuncy bin j
        fJ = frequency of the spectrum at frequency bin j

    * Input:
        * rawEMGPowerSpectrum: PSD as list
        * frequencies: frequencies of the PSD spectrum as list
    * Output:
        * Mean Frequency of the PSD

    :param rawEMGPowerSpectrum: power spectrum of the EMG signal
    :type rawEMGPowerSpectrum: list
    :param frequencies: frequencies of the PSD
    :type frequencies: list
    :return: mean frequency of the EMG power spectrum
    :rtype: float
    """
    a = []
    for i in range(0, len(frequencies)):
        a.append(frequencies[i] * rawEMGPowerSpectrum[i])
    b = sum(rawEMGPowerSpectrum)
    MNF = sum(a) / b
    return MNF


def calc_freq_chunks(data, time, nPts, sampling_rate):
    """
    Calculate the Freq Mean/Median for each chunk of nPts
        It returns time and freq arrays whose lengths are = data/nPts.
    """

    # Divide data into chunks of length nPts
    data_chunks = list(divide_chunks(data, nPts))
    time_chunks = list(divide_chunks(time, nPts))

    # Loop through each chunk and calculate FFT
    med_freq_chunks_result = []
    mean_freq_chunks_result = []
    for x in data_chunks:
        f, fft = calculate_welch_spectrum(x, sampling_rate)
        med = mean_freq(fft, f)
        mean = median_freq(fft, f)
        med_freq_chunks_result.append(med)
        mean_freq_chunks_result.append(mean)

    # Loop through time and calculate average time that RMS corresponds to
    time_chunks_result = []
    for x in time_chunks:
        time_chunks_result.append(np.mean(x))

    # conver to np array
    med_freq_chunks_result = np.array(med_freq_chunks_result)
    mean_freq_chunks_result = np.array(mean_freq_chunks_result)
    time_chunks_result = np.array(time_chunks_result)

    return mean_freq_chunks_result, med_freq_chunks_result, time_chunks_result


###############################################################################
#                                                                             #
#                                RMS FEATURES                                 #
#                                                                             #
###############################################################################
def calc_rms(data):
    """
    RMS calculation
    """
    mean = float(np.mean(data))
    n = len(data)
    sum_squares = 0
    for i in data:
        sum_squares += (i - mean) ** 2
    rms = np.sqrt(sum_squares / n)
    return rms


def calc_rms_chunks(data, time, nPts):
    """
    Calculate the RMS for each chunk of nPts
        It returns time and rms arrays whose lengths are = data/nPts.
    """

    # Divide data into chunks of length nPts
    data_chunks = list(divide_chunks(data, nPts))
    time_chunks = list(divide_chunks(time, nPts))

    # Loop through each chunk and calculate RMS
    rms_chunks_result = []
    for x in data_chunks:
        rms_chunks_result.append(calc_rms(x))

    # Loop through time and calculate average time that RMS corresponds to
    time_chunks_result = []
    for x in time_chunks:
        time_chunks_result.append(np.mean(x))

    # conver to np array
    rms_chunks_result = np.array(rms_chunks_result)
    time_chunks_result = np.array(time_chunks_result)

    return rms_chunks_result, time_chunks_result


def divide_chunks(arr, n):
    """
    Divides an array of length l into chunks of length n
    """
    # looping till length l
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


def moving_avg_rms(data, window_size=301):
    """
    Moving avg RMS
        https://stackoverflow.com/questions/8245687/numpy-root-mean-squared-rms-smoothing-of-a-signal
        NOTE: len(rms) = len(data) - (window_size - 1)  Make sure to account for this
        in plotting plot(time[window_size-1::], rms)
    """
    data2 = np.power(data, 2)
    window = np.ones(window_size) / float(window_size)
    rms = np.sqrt(np.convolve(data2, window, 'valid'))
    return rms


###############################################################################
#                                                                             #
#                          TIME DOMAIN FEATURES                               #
#                                                                             #
###############################################################################


def find_peaks_algo1(data):
    """
    Algo 1: Find Peaks and Troughs
        This algorithm finds the indices of peaks and troughs in an impedance
        measurement.

        Input:
            + data:    1D array of impedance data (units: can be volts or resistance)

        Outputs:
            + peaks:   1D array of indices corresponding to peaks in impedance data
            + troughs: 1D array of indices corresponding to troughs in impedance data
    """
    peaks, _ = signal.find_peaks(
        data, height=(None, None), prominence=(None, None), threshold=(None, None)
    )
    troughs, _ = signal.find_peaks(-1 * data)

    return peaks, troughs


def find_heights_algo1(time, data, peaks, troughs):
    """
    Algo 1: Calculate Impedance with FLIP Method
        This algorithm finds the peak-to-peak height of each square wave in an
        impedance measurement.

        Goes with find_peaks_algo1

        Inputs:
            + data:    1D array of impedance data (units: can be volts or resistance)
            + peaks:   1D array of indices corresponding to peaks in impedance data
            + troughs: 1D array of indices corresponding to troughs in impedance data

        Outputs:
            + heights:  1D array of the height of each square wave
                        (units: same unit of input data)
            + avg_time: 1D time vector corresponding heights.
                        The middle btw each peak and trough occurance.
                        (units: same unit as input time)
            + avg_amps: 1D array of mean data at each avg_time.
                        The mean value of each square wave.
                        (units: same unit of input data)

        Limitation of this algo:
            It does not handle unevenness of square waves well in impedance measurements
            where there is baseline drift
            (either due to movement or the inherent RC constant).
            The height of neighboring square waves might be drastically different due to
            uneven timing in these datasets.
            Heights will generally be larger than with find_heights_algo2.

    """
    # if trough comes first: remove 1st index so peak is first
    # (NOTE: not sure this matters so prob delete in future)
    if peaks[0] > troughs[0]:
        troughs = troughs[1:]
    pairs = list(zip(peaks, troughs))
    print('ALGO 1 types', type(peaks), type(troughs))
    print('data', type(data))
    heights = []
    avg_time = []
    avg_amps = []
    for x in pairs:
        # print('ALGO 1type(x)', type(x[0]), type(x[1]) )
        heights.append(abs(data[x[0]] - data[x[1]]))  # peak is first in tuple
        avg_time.append(np.mean([time[x[0]], time[x[1]]]))
        avg_amps.append(np.mean([data[x[0]], data[x[1]]]))
    return heights, avg_time, avg_amps


def find_peaks_algo2(data):
    """
    Algo 2: Find Peaks and Troughs
        This algorithm finds the indices of peaks and troughs in an impedance
        measurement by first taking the derivative of the data and then finding
        the peaks/troughs (correspondto points of greatest slope).

        Input:
            + data:    1D array of impedance data (units: can be volts or resistance)

        Outputs:
            + peaks:   1D array of indices corresponding to peaks in impedance data
            + troughs: 1D array of indices corresponding to troughs in impedance data


        NOTE: will need to play around with prominence/height/treshold variables as more
        data is collected.
    """
    data_1st_deriv = np.gradient(np.gradient(data))
    peaks, _ = signal.find_peaks(
        data_1st_deriv, height=100, prominence=(None, None), threshold=(None, None)
    )
    troughs, _ = signal.find_peaks(-1 * data_1st_deriv, height=100)

    return peaks, troughs


def find_heights_algo2(time, data, peaks, troughs):
    """
    Algo 1: Calculate Impedance with FLIP Method
        This algorithm finds the peak-to-peak height of each square wave in an
        impedance measurement.

        Goes with find_peaks_algo2

        Inputs:
            + data:    1D array of impedance data (units: can be volts or resistance)
            + peaks:   1D array of indices corresponding to peaks in impedance data
            + troughs: 1D array of indices corresponding to troughs in impedance data

        Outputs:
            + heights:  1D array of the height of each square wave
                        (units: same unit of input data)
            + avg_time: 1D time vector corresponding heights.
            The middle btw each peak and trough occurance.
                        (units: same unit as input time)
            + avg_amps: 1D array of mean data at each avg_time.
            The mean value of each square wave.
                        (units: same unit of input data)

        Limitation of this algo:
            It is more unreliable (does not always find the exact peak/trough) than
            find_peaks_algo1 but more correct to calculate height of each edge of
            the square wave. It calculates height of each edge of square wave so there
            is more data about the square wave to average together. Heights will
            generally be smaller than those found with find_heights_algo1.


        NOTE: In future, combine the two find_heights_algo[n] functions b/c they are
        the same other than initial processing.
    """
    pairs = list(zip(peaks, troughs))
    heights = []
    avg_time = []
    avg_amps = []
    for x in pairs:
        heights.append(abs(data[x[0]] - data[x[1]]))  #
        avg_time.append(np.mean([time[x[0]], time[x[1]]]))
        avg_amps.append(np.mean([data[x[0]], data[x[1]]]))
    return heights, avg_time, avg_amps


###############################################################################
#                                                                             #
#                              FILTERS                                        #
#                                                                             #
###############################################################################
def butter_highpass(filter_order, cutoff, sampling_freq):
    b, a = signal.butter(
        filter_order, cutoff, btype='highpass', analog=False, fs=sampling_freq
    )
    # scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
    # Numerator (b) and denominator (a) polynomials of the IIR filter
    return b, a


def butter_highpass_filter(data, filter_order, cutoff, sampling_freq):
    b, a = butter_highpass(filter_order, cutoff, sampling_freq)
    y = signal.filtfilt(b, a, data)  # signal.filtfilt vs signal.lfilter?
    # y = pd.Series(y) #keep y as a Pandas Series like it was before filter
    return y  # <-- output is a numpy array


###############################################################################
#                                                                             #
#                         DATA PROCESSING                                     #
#                                                                             #
###############################################################################
def processRawData(time_raw_us, data_bits, v_ref, channel_gain):
    """
    Convert data from bits to uV, and time from us to s
    input: time_raw_us = millisec (numpy array or pandas series)
           data_bits = bits (raw output from CDE file) (numpy array or pandas series)
    output: time_sec (numpy array or pandas series, whichever the input was)
            data_uV (numpy array or pandas series, whichever the input was)
    """
    print('---DSP: PROCESSRAWDATA---')

    # Calculate Time array (starts at 0)
    time_sec = (time_raw_us - time_raw_us[0]) / 1000000.0

    # Convert data from bits to voltage
    data_uV = (
        1e6 * data_bits * v_ref / (channel_gain * (2**23 - 1))
    )  # units: microvolts (uV)

    # Return time (sec) and data (uV)
    # these are pandas Series (RMS and FFT data are numpy arrays).
    # Should probably pick one format and convert all others to it.
    return (
        time_sec,
        data_uV,
    )


def setGain(which_channel, gainlevel, isRLD):
    """
    Calculate channel gain by accounting for gainlevel (i.e. gain set in the INIT.f)
    and channel properties (i.e. built-in gain)
    Adjusts when it's an RLD channel
    which_channel: 'c1', 'c2_nomux', 'c2_mux5x', 'c2_mux1x', 'c4', 'c5'
    gainlevel = 0-12
    isRLD = 0 or 1
    """
    print('---DSP: SETGAIN---')
    if which_channel == 'c2_nomux':
        if isRLD == 1:
            channel_gain = (
                gainlevel  # ***** NEED TO CONFIRM THIS BASED ON EXPERIMENTS *******
            )
        else:
            Rg = 4.84  # Only for Channel 2 --> MUX 5x: 4.87, MUX 1x: 200, No MUX: 4.84
            ina_gain = 1 + 19.8 / Rg
            channel_gain = gainlevel * ina_gain
    elif which_channel == 'c2_mux5x':
        if isRLD == 1:
            channel_gain = gainlevel
        else:
            Rg = 4.87  # Only for Channel 2 --> MUX 5x: 4.87, MUX 1x: 200, No MUX: 4.84
            ina_gain = 1 + 19.8 / Rg
            channel_gain = gainlevel * ina_gain
    elif which_channel == 'c2_mux1x':
        if isRLD == 1:
            channel_gain = gainlevel
        else:
            Rg = 200  # Only for Channel 2 --> MUX 5x: 4.87, MUX 1x: 200, No MUX: 4.84
            ina_gain = 1 + 19.8 / Rg
            channel_gain = gainlevel * ina_gain
    elif which_channel == 'c1':
        channel_gain = gainlevel
    elif (which_channel == 'c4') | (which_channel == 'c5'):
        channel_gain = gainlevel * 100
    else:
        print('This is not a valid channel')  # ***** MAKE GOOD ERROR MESSAGE *****
        return

    if isRLD == 1:
        print(
            "This in an RLD channel. Ignore its actual name, which is: ",
            which_channel,
            ". The gain is now: ",
            channel_gain,
        )
    else:
        print("The channel is: ", which_channel, ". The gain is now: ", channel_gain)

    return channel_gain


###############################################################################
#                                                                             #
#                             DATA CHECKS                                     #
#                                                                             #
###############################################################################
def check_for_missing(dataframe, sampling_freq, readhex=1):
    print('---DSP: CHECK_FOR_MISSING---')

    # Check if the dataframe has any missing samples
    #  NOTE: THIS CHECKING OF EMPTY DATAFRAME SHOULD BE COMBINED WITH PLOTPREP()
    if (not isinstance(dataframe, pd.DataFrame)) and dataframe == []:
        print('Dataframe is empty')
    else:
        # Pull out time
        time = dataframe.time / 1000000.0  # units: seconds (s)

        # -------- Time Array -----------
        time_intervals = 1 / sampling_freq  # units: seconds (s)
        calc_diffs_btw_timestamps = np.array([t - s for s, t in zip(time, time[1:])])
        where_timestamps_notExpected = np.where(
            (
                (time_intervals - 1 > calc_diffs_btw_timestamps)
                | (calc_diffs_btw_timestamps > time_intervals + 1)
            )
        )  # & (calc_diffs_btw_timestamps != 3))[0]
        #  NOTE: Maybe don't need the +/-1ms from the expected time interval (?)
        where_timestamps_notExpected = where_timestamps_notExpected[0]
        number_of_millisec_skipped = {}
        for var in where_timestamps_notExpected:
            number_of_millisec_skipped[
                str(str(time[var] - time[0]) + 'ms, ' + 'Row #' + str(var))
            ] = (str(calc_diffs_btw_timestamps[var]) + 'ms skipped')
        # NOTE: If there is NaN in the time_intervals then that needs to be reported.
        # There is a "165221.0, NaN, 190710.0" that doesn't get picked up by code.
        # print('calc_diffs_btw_timestamps',calc_diffs_btw_timestamps[0:50])

        if np.array(where_timestamps_notExpected).size > 0:
            print(" ")
            print("*** IMPORTANT ***")
            print("There are missing time stamps at: ", number_of_millisec_skipped)
        else:
            print(" ")
            print("*** IMPORTANT ***")
            print("There are no missing time stamps found. ")

        # -------- ID Array ------------
        if readhex == 0:
            ids = dataframe.id
            calc_diffs_btw_ids = np.array([t - s for s, t in zip(ids, ids[1:])])
            where_ids_not1 = np.where(
                calc_diffs_btw_ids != 1
            )  # see if any are not == 1
            if np.array(where_ids_not1).size > 0:
                print(" ")
                print("*** IMPORTANT ***")
                print(
                    "There are missing ID counts after the following IDs: {}".format(
                        where_ids_not1
                    )
                )
