import numpy as np

# response functions for neuront types from "Hippocampal Encoding of Non-Spatial Trace Conditioning"

# Define the timing of stimuli in seconds
CS_duration = 0.25  # 250 ms
US_start_time = CS_duration + 0.5  # 500 ms after the end of CS
US_duration = 0.100

def gaussian_peak(time, magnitude, peak_time, sd, duration):
    CS_duration = 0.25  # 250 ms
    US_start_time = CS_duration + 0.5  # 500 ms after the end of CS
    US_duration = 0.100
    response = magnitude * np.exp(-((time - peak_time) ** 2) / (2 * sd ** 2))
    # Ensure that `time` is an array to support item assignment
    if isinstance(time, np.ndarray):
        response[time > (peak_time + duration)] = 0
    else:
        # If `time` is a single value, use a conditional to determine the response
        response = 0 if time > (peak_time + duration) else response
    return response

# Redefine bimodal_response to decay back to baseline after CS and US
def bimodal_response(time, magnitudes, peak_times, sds, baseline, cs_duration, us_duration):
    CS_duration = 0.25  # 250 ms
    US_start_time = CS_duration + 0.5  # 500 ms after the end of CS
    US_duration = 0.100
    peak1 = gaussian_peak(time, magnitudes[0], peak_times[0], sds[0], cs_duration)
    peak2 = gaussian_peak(time, magnitudes[1], peak_times[1], sds[1], us_duration)
    # Ensure the response is an array if `time` is an array
    total_response = peak1 + peak2 + baseline if isinstance(time, np.ndarray) else peak1 + peak2 + baseline
    return total_response

# Redefine linear_decay_response to decay after peak
def linear_decay_response(time, start, peak_time, end, max_value, baseline):
    CS_duration = 0.25  # 250 ms
    US_start_time = CS_duration + 0.5  # 500 ms after the end of CS
    US_duration = 0.100
    if time < peak_time:
        return max_value
    elif peak_time <= time <= end:
        slope = (baseline - max_value) / (end - peak_time)
        return slope * (time - peak_time) + max_value
    else:
        return baseline

# Define the response profiles for each cell type with estimated baselines and adjusted times
# dividing by 2400 bc 80 trials and 7.5 samples per second
response_profiles = {
    1: {'response_func': lambda t: bimodal_response(t, magnitudes=[15/2400, 40/2400], peak_times=[0.1, US_start_time], sds=[0.05, 0.1], baseline=10/2400, cs_duration=CS_duration, us_duration=US_duration), 'baseline': 10/2400},
    2: {'response_func': lambda t: gaussian_peak(t, magnitude=20/2400, peak_time=US_start_time, sd=0.15, duration=US_duration) + 5/2400, 'baseline': 5/2400},
    3: {'response_func': lambda t: linear_decay_response(t, start=0, peak_time=0.1 + CS_duration, end=US_start_time, max_value=25/2400, baseline=25/2400), 'baseline': 25/2400},
    4: {'response_func': lambda t: linear_decay_response(t, start=0, peak_time=0.1 + CS_duration, end=US_start_time, max_value=20/2400, baseline=20/2400), 'baseline': 20/2400},
    5: {'response_func': lambda t: linear_decay_response(t, start=0, peak_time=US_start_time, end=US_start_time + US_duration, max_value=30/2400, baseline=10/2400), 'baseline': 10/2400},
    6: {'response_func': lambda t: gaussian_peak(t, magnitude=15/2400, peak_time=US_start_time, sd=0.1, duration=US_duration) + 15/2400, 'baseline': 15/2400},
    7: {'response_func': lambda t: bimodal_response(t, magnitudes=[10/2400, 30/2400], peak_times=[0.1, US_start_time], sds=[0.05, 0.15], baseline=10/2400, cs_duration=CS_duration, us_duration=US_duration), 'baseline': 10/2400},
    8: {'response_func': lambda t: linear_decay_response(t, start=0, peak_time=0.1, end=0.1 + CS_duration, max_value=10/2400, baseline=10/2400), 'baseline': 10/2400}
}
