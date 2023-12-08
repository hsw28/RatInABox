import numpy as np

# response functions for neuront types from "Hippocampal Encoding of Non-Spatial Trace Conditioning"

# Define response functions with an additional 'last_fr' parameter for the baseline

def gaussian_peak(time, magnitude, peak_time, sd, last_fr):
    return magnitude * np.exp(-((time - peak_time) ** 2) / (2 * sd ** 2)) + last_fr

def bimodal_response(time, magnitudes, peak_times, sds, last_fr):
    peak1 = gaussian_peak(time, magnitudes[0], peak_times[0], sds[0], last_fr)
    peak2 = gaussian_peak(time, magnitudes[1], peak_times[1], sds[1], last_fr)
    return peak1 + peak2 + last_fr

def linear_response(time, start, end, start_value, end_value, last_fr):
    if start <= time <= end:
        slope = (end_value - start_value) / (end - start)
        return slope * (time - start) + start_value + last_fr
    else:
        return last_fr

def uniform_response(time, last_fr):
    return last_fr

# Define the response profiles for each cell type with estimated baselines and adjusted times
response_profiles = {
    1: {'response_func': lambda t, last_fr: bimodal_response(t, magnitudes=[15/80, 40/80], peak_times=[0.3, 0.8], sds=[0.05, 0.1], last_fr=last_fr)},
    2: {'response_func': lambda t, last_fr: gaussian_peak(t, magnitude=25/80, peak_time=0.6, sd=0.15, last_fr=last_fr)},
    3: {'response_func': lambda t, last_fr: linear_response(t, start=0, end=0.6, start_value=25/80, end_value=0, last_fr=last_fr)},
    4: {'response_func': lambda t, last_fr: linear_response(t, start=0, end=0.6, start_value=20/80, end_value=0.05, last_fr=last_fr)},
    5: {'response_func': lambda t, last_fr: linear_response(t, start=0, end=0.6, start_value=10/80, end_value=30/80, last_fr=last_fr)},
    6: {'response_func': lambda t, last_fr: uniform_response(t, last_fr=last_fr)},
    7: {'response_func': lambda t, last_fr: bimodal_response(t, magnitudes=[10, 25], peak_times=[0.3, 0.8], sds=[0.05, 0.1], last_fr=last_fr)},
    8: {'response_func': lambda t, last_fr: linear_response(t, start=0, end=0.6, start_value=10/80, end_value=0, last_fr=last_fr)}
}
