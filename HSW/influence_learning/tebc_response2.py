import numpy as np

# Adjusted response functions considering 10 ms bins

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
    # Bimodal response, lower first peak and higher second peak
    1: {'response_func': lambda t: bimodal_response(t, magnitudes=[0.15, 0.4], peak_times=[0.3, 0.8], sds=[0.05, 0.1], baseline=0.1)},
    # Single peak with fast rise and slow fall
    2: {'response_func': lambda t: gaussian_peak(t, magnitude=0.25, peak_time=0.6, sd=0.15) + 0.05},
    # Sharp decrease from baseline
    3: {'response_func': lambda t: linear_response(t, start=0, end=0.6, start_value=0.25, end_value=0) + 0.25},
    # Slow decrease, quick rise, and slow fall
    4: {'response_func': lambda t: linear_response(t, start=0, end=0.6, start_value=0.2, end_value=0.05) + 0.2},
    # Sharp increase and linear descent
    5: {'response_func': lambda t: linear_response(t, start=0, end=0.6, start_value=0.1, end_value=0.3) + 0.1},
    # Uniform response close to baseline
    6: {'response_func': lambda t: uniform_response(t, baseline=0.15)},
    # Similar to type 1 but with less pronounced peaks
    7: {'response_func': lambda t: bimodal_response(t, magnitudes=[0.1, 0.25], peak_times=[0.3, 0.8], sds=[0.05, 0.1], baseline=0.1)},
    # Gradual decrease from baseline
    8: {'response_func': lambda t: linear_response(t, start=0, end=0.6, start_value=0.1, end_value=0) + 0.1}
}
