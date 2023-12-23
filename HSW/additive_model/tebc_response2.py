import numpy as np

# Define the timing of stimuli in seconds
CS_duration = 0.25  # 250 ms
US_start_time = CS_duration + 0.5  # 500 ms after the end of CS
US_duration = 0.100

def gaussian_peak(time_since_CS, magnitude, peak_time, sd, duration, baseline):
    response = magnitude * np.exp(-((time_since_CS - peak_time) ** 2) / (2 * sd ** 2))
    return np.where(time_since_CS > (peak_time + duration), baseline, response)

def bimodal_response(time_since_CS, magnitudes, peak_times, sds, baseline, cs_duration, us_duration):
    peak1 = gaussian_peak(time_since_CS, magnitudes[0], peak_times[0], sds[0], cs_duration, baseline)
    peak2 = gaussian_peak(time_since_CS, magnitudes[1], peak_times[1], sds[1], us_duration, baseline)
    return peak1 + peak2

def bimodal_response2(time_since_CS, magnitudes, peak_times, sds, baseline, cs_duration, us_duration, noise_level):
    broadened_peak1 = gaussian_peak(time_since_CS, magnitudes[0], peak_times[0], sds[0] * 1.5, cs_duration, baseline)
    broadened_peak2 = gaussian_peak(time_since_CS, magnitudes[1], peak_times[1], sds[1] * 1.5, us_duration, baseline)
    total_response = broadened_peak1 + broadened_peak2
    total_response += np.random.normal(0, noise_level)
    return total_response

def linear_decay_response(time_since_CS, peak_time, end, max_value, baseline):
    if time_since_CS < peak_time:
        response = max_value
    elif peak_time <= time_since_CS <= end:
        slope = (baseline - max_value) / (end - peak_time)
        response = slope * (time_since_CS - peak_time) + max_value
    else:
        response = baseline
    return response

def noisy_uniform_response(time_since_CS, baseline, noise_level=0.005):
    CS_duration = 0.25  # 250 ms
    US_start_time = CS_duration + 0.5  # 500 ms after the end of CS
    US_duration = 0.100
    return baseline + np.random.normal(0, noise_level)

# Response profiles for each cell type
response_profiles = {
    1: {'response_func': lambda t, baseline: bimodal_response(t, [5/2400, 20/2400], [0.1, US_start_time], [0.1, 0.1], baseline, CS_duration, US_duration), 'baseline': 10/2400},
    2: {'response_func': lambda t, baseline: gaussian_peak(t, 20/2400, US_start_time, 0.15, US_duration, baseline) + 5/2400, 'baseline': 5/2400},
    3: {'response_func': lambda t, baseline: linear_decay_response(t, 0.1 + CS_duration, US_start_time, 25/2400, baseline), 'baseline': 25/2400},
    4: {'response_func': lambda t, baseline: linear_decay_response(t, 0, 0.1 + CS_duration, US_start_time, 20/2400, baseline), 'baseline': 20/2400},
    5: {'response_func': lambda t, baseline: linear_decay_response(t, 0, US_start_time, US_start_time + US_duration, 30/2400, baseline), 'baseline': 10/2400},
    6: {'response_func': lambda t, baseline: noisy_uniform_response(t, baseline), 'baseline': 15/2400},
    7: {'response_func': lambda t, baseline: bimodal_response2(t, [8/2400, 20/2400], [0.1, US_start_time], [0.08, 0.15], baseline, CS_duration, US_duration, 0.001), 'baseline': 10/2400},
    8: {'response_func': lambda t, baseline: linear_decay_response(t, 0, 0.1, 0.1 + CS_duration, 10/2400, baseline), 'baseline': 10/2400}
}
