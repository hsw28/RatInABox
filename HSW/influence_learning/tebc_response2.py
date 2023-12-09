import numpy as np

# response functions for neuront types from "Hippocampal Encoding of Non-Spatial Trace Conditioning"

def gaussian_peak(time, magnitude, peak_time, sd):
    return magnitude * np.exp(-((time - peak_time) ** 2) / (2 * sd ** 2))

def bimodal_response(time, magnitudes, peak_times, sds, baseline):
    peak1 = gaussian_peak(time, magnitudes[0], peak_times[0], sds[0])
    peak2 = gaussian_peak(time, magnitudes[1], peak_times[1], sds[1])
    return peak1 + peak2 + baseline

def linear_response(time, start, end, start_value, end_value):
    if start <= time <= end:
        slope = (end_value - start_value) / (end - start)
        return slope * (time - start) + start_value
    else:
        return start_value if time < start else end_value

def uniform_response(time, baseline):
    return baseline

# Define the response profiles for each cell type with estimated baselines and adjusted times
response_profiles = {
    1: {'response_func': lambda t: bimodal_response(t, magnitudes=[15/80, 40/80], peak_times=[0.3, 0.8], sds=[0.05, 0.1], baseline=10/80), 'baseline': 10/80},
    2: {'response_func': lambda t: gaussian_peak(t, magnitude=2/80, peak_time=0.6, sd=0.15) + 5/80, 'baseline': 5/80},
    3: {'response_func': lambda t: linear_response(t, start=0, end=0.6, start_value=25/80, end_value=0) + 25/80, 'baseline': 25/80},
    4: {'response_func': lambda t: linear_response(t, start=0, end=0.6, start_value=20/80, end_value=0.05) + 20/80, 'baseline': 20/80},
    5: {'response_func': lambda t: linear_response(t, start=0, end=0.6, start_value=10/80, end_value=30/80) + 10/80, 'baseline': 10/80},
    6: {'response_func': lambda t: uniform_response(t, baseline=15/80), 'baseline': 15/80},
    7: {'response_func': lambda t: bimodal_response(t, magnitudes=[10/80, 25/80], peak_times=[0.3, 0.8], sds=[0.05, 0.1], baseline=10/80), 'baseline': 10/80},
    8: {'response_func': lambda t: linear_response(t, start=0, end=0.6, start_value=10/80, end_value=0) + 10/80, 'baseline': 10/80}
}
