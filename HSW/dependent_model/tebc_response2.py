import numpy as np

def gaussian_peak(time, magnitude_factor, peak_time, sd, baseline):
    return baseline * (1 + magnitude_factor * np.exp(-((time - peak_time) ** 2) / (2 * sd ** 2))) - baseline

def bimodal_response(time, magnitude_factors, peak_times, sds, baseline):
    peak1 = gaussian_peak(time, magnitude_factors[0], peak_times[0], sds[0], baseline)
    peak2 = gaussian_peak(time, magnitude_factors[1], peak_times[1], sds[1], baseline)
    return peak1 + peak2 - baseline  # Adjust for the double addition of baseline

def linear_response(time, start, end, start_factor, end_factor, baseline):
    if start <= time <= end:
        slope = (end_factor - start_factor) / (end - start)
        return baseline * (1 + slope * (time - start) + start_factor) - baseline
    else:
        return baseline * (1 + start_factor if time < start else end_factor)- baseline

def uniform_response(time, baseline):
    return baseline- baseline

# Define the response profiles for each cell type
response_profiles = {
    1: {'response_func': lambda t, baseline: bimodal_response(t, magnitude_factors=[15, 40], peak_times=[0.3, 0.8], sds=[0.05, 0.1], baseline=baseline)},
    2: {'response_func': lambda t, baseline: gaussian_peak(t, magnitude_factor=2, peak_time=0.6, sd=0.15, baseline=baseline)},
    3: {'response_func': lambda t, baseline: linear_response(t, start=0, end=0.6, start_factor=1, end_factor=0.17, baseline=baseline)},
    4: {'response_func': lambda t, baseline: linear_response(t, start=0, end=0.6, start_factor=1, end_factor=1.05/20, baseline=baseline)},
    5: {'response_func': lambda t, baseline: linear_response(t, start=0, end=0.6, start_factor=1, end_factor=30/10, baseline=baseline)},
    6: {'response_func': lambda t, baseline: uniform_response(t, baseline=baseline)},
    7: {'response_func': lambda t, baseline: bimodal_response(t, magnitude_factors=[10, 25], peak_times=[0.3, 0.8], sds=[0.05, 0.1], baseline=baseline)},
    8: {'response_func': lambda t, baseline: linear_response(t, start=0, end=0.6, start_factor=1, end_factor=0, baseline=baseline)}
}
