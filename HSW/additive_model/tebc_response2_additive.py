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
# dividing by 600 bc 80 trials and 7.5 samples per second
response_profiles = {
    1: {'response_func': lambda t: bimodal_response(t, magnitudes=[15/600, 40/600], peak_times=[0.3, 0.8], sds=[0.05, 0.1], baseline=10/600), 'baseline': 10/600},
    2: {'response_func': lambda t: gaussian_peak(t, magnitude=2/600, peak_time=0.6, sd=0.15) + 5/600, 'baseline': 5/600},
    3: {'response_func': lambda t: linear_response(t, start=0, end=0.6, start_value=25/600, end_value=0) + 25/600, 'baseline': 25/600},
    4: {'response_func': lambda t: linear_response(t, start=0, end=0.6, start_value=20/600, end_value=0.05) + 20/600, 'baseline': 20/600},
    5: {'response_func': lambda t: linear_response(t, start=0, end=0.6, start_value=10/600, end_value=30/600) + 10/600, 'baseline': 10/600},
    6: {'response_func': lambda t: uniform_response(t, baseline=15/600), 'baseline': 15/600},
    7: {'response_func': lambda t: bimodal_response(t, magnitudes=[10/600, 25/600], peak_times=[0.3, 0.8], sds=[0.05, 0.1], baseline=10/600), 'baseline': 10/600},
    8: {'response_func': lambda t: linear_response(t, start=0, end=0.6, start_value=10/600, end_value=0) + 10/600, 'baseline': 10/600}
}
