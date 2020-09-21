from scipy.signal import butter,filtfilt

from numba import jit
import bisect

def is_number_in_sorted_vector(sorted_vector, num):
    index = bisect.bisect_left(sorted_vector, num)
    return index != len(sorted_vector) and sorted_vector[index] == num

# def butter_lowpass(cutoff, fs, order=5):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return b, a


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

@jit
def first_order_lowpass_filter(signal_in, signal_out, tau, dt):

    alpha_lowpass = dt / (tau + dt)

    signal_out[0] = signal_in[0]

    for i in range(1, len(signal_in)):
        signal_out[i] = alpha_lowpass*signal_in[i] + (1-alpha_lowpass)*signal_out[i-1]
