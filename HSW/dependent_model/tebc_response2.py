import numpy as np

# Define the timing of stimuli in seconds
CS_duration = 0.25  # 250 ms
US_start_time = CS_duration + 0.5  # 500 ms after the end of CS
US_duration = 0.100

def gaussian_peak(time, magnitude, peak_time, sd, duration, baseline):
    CS_duration = 0.25  # 250 ms
    US_start_time = CS_duration + 0.5  # 500 ms after the end of CS
    US_duration = 0.100
    response = magnitude * np.exp(-((time - peak_time) ** 2) / (2 * sd ** 2))
    # Subtract the baseline to ensure that the response returns to the baseline level after the peak
    response = np.where(time > (peak_time + duration), baseline, response)
    return response - baseline

def bimodal_response(time, magnitudes, peak_times, sds, baseline, cs_duration, us_duration):
    CS_duration = 0.25  # 250 ms
    US_start_time = CS_duration + 0.5  # 500 ms after the end of CS
    US_duration = 0.100
    peak1 = gaussian_peak(time, magnitudes[0], peak_times[0], sds[0], cs_duration, baseline)
    peak2 = gaussian_peak(time, magnitudes[1], peak_times[1], sds[1], us_duration, baseline)
    # The baseline is already considered in the gaussian_peak function
    return peak1 + peak2 - baseline   # Subtract baseline to ensure peaks are relative to it

def linear_decay_response(time, start, peak_time, end, max_value, baseline):
    CS_duration = 0.25  # 250 ms
    US_start_time = CS_duration + 0.5  # 500 ms after the end of CS
    US_duration = 0.10
    # Handle the case where 'time' is a single integer value
    if isinstance(time, int) or isinstance(time, float):
        if time < peak_time:
            return max_value- baseline
        elif peak_time <= time <= end:
            slope = (baseline - max_value) / (end - peak_time)
            return slope * (time - peak_time) + max_value- baseline
        else:
            return baseline - baseline

    # Handle the case where 'time' is a numpy array
    else:
        response = np.full_like(time, baseline, dtype=float)
        slope = (baseline - max_value) / (end - peak_time)
        decay_mask = (time >= peak_time) & (time <= end)
        response[decay_mask] = slope * (time[decay_mask] - peak_time) + max_value
        return response- baseline


# Define the response profiles for each cell type
response_profiles = {
    1: {'response_func': lambda t, baseline: bimodal_response(t, magnitudes=[1.5*baseline, 4*baseline], peak_times=[0.1, US_start_time], sds=[0.05, 0.1], baseline=baseline, cs_duration=CS_duration, us_duration=US_duration)},
    2: {'response_func': lambda t, baseline: gaussian_peak(t, magnitude=4*baseline, peak_time=US_start_time, sd=0.15, duration=US_duration, baseline=baseline) + baseline},
    3: {'response_func': lambda t, baseline: linear_decay_response(t, start=0, peak_time=0.1 + CS_duration, end=US_start_time, max_value=baseline, baseline=baseline)},
    4: {'response_func': lambda t, baseline: linear_decay_response(t, start=0, peak_time=0.1 + CS_duration, end=US_start_time, max_value=baseline, baseline=baseline)},
    5: {'response_func': lambda t, baseline: linear_decay_response(t, start=0, peak_time=US_start_time, end=US_start_time + US_duration, max_value=3*baseline, baseline=baseline)},
    6: {'response_func': lambda t, baseline: gaussian_peak(t, magnitude=baseline, peak_time=US_start_time, sd=0.1, duration=US_duration, baseline=baseline) + baseline},
    7: {'response_func': lambda t, baseline: bimodal_response(t, magnitudes=[baseline, 3*baseline], peak_times=[0.1, US_start_time], sds=[0.05, 0.15], baseline=baseline, cs_duration=CS_duration, us_duration=US_duration)},
    8: {'response_func': lambda t, baseline: linear_decay_response(t, start=0, peak_time=0.1, end=0.1 + CS_duration, max_value=baseline, baseline=baseline)}
}
