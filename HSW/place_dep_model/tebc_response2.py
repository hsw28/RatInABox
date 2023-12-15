import numpy as np

# Constants for timings

def type_one_response(time_since_CS, baseline): #in place field fast firing, moving
    CS_duration = 0.25  # CS duration in seconds
    CS_to_US_delay = 0.5  # Delay between CS and US in seconds
    US_time = CS_duration + CS_to_US_delay  # Time when US occurs
    total_time = CS_duration + CS_to_US_delay + 0.25  # Total duration considered for response

    if time_since_CS < CS_duration:
        # Firing rate increases by 50% at the start of CS
        return baseline * 1.5
    elif CS_duration <= time_since_CS < (CS_duration + CS_to_US_delay):
        # Slow decrease after CS until US
        return baseline * (1.5 - 0.5 * (time_since_CS - CS_duration) / CS_to_US_delay)
    elif time_since_CS == (CS_duration + CS_to_US_delay):
        # Sharp increase at US
        return baseline * 2
    else:
        # Return to baseline after US
        return baseline

def type_two_response(time_since_CS, baseline): #in place field medium firing, moving
    CS_duration = 0.25  # CS duration in seconds
    CS_to_US_delay = 0.5  # Delay between CS and US in seconds
    US_time = CS_duration + CS_to_US_delay  # Time when US occurs
    total_time = CS_duration + CS_to_US_delay + 0.25  # Total duration considered for response
    if time_since_CS < (CS_duration + CS_to_US_delay):
        # No change during CS
        return baseline
    elif time_since_CS == (CS_duration + CS_to_US_delay):
        # Increase at US
        return baseline * 1.2
    else:
        # Remain at increased rate after US
        return baseline * 1.2

def type_three_response(time_since_CS, baseline): #out of place field but moving
    CS_duration = 0.25  # CS duration in seconds
    CS_to_US_delay = 0.5  # Delay between CS and US in seconds
    US_time = CS_duration + CS_to_US_delay  # Time when US occurs
    total_time = CS_duration + CS_to_US_delay + 0.25  # Total duration considered for response
    if time_since_CS < CS_duration:
        # Sharp decrease at CS
        return baseline * 0.5
    elif CS_duration <= time_since_CS < (CS_duration + CS_to_US_delay):
        # Remain at decreased rate until US
        return baseline * 0.5
    elif time_since_CS == (CS_duration + CS_to_US_delay):
        # Slight increase at US
        return baseline * 0.6
    else:
        # Return to baseline after US
        return baseline

def type_four_response(time_since_CS, baseline): #not moving, in field
    CS_duration = 0.25  # CS duration in seconds
    CS_to_US_delay = 0.5  # Delay between CS and US in seconds
    US_time = CS_duration + CS_to_US_delay  # Time when US occurs
    total_time = CS_duration + CS_to_US_delay + 0.25  # Total duration considered for response
    if time_since_CS < CS_duration:
        # Sudden increase at the start of CS
        return baseline * 1.5
    elif CS_duration <= time_since_CS < US_time:
        # Gradual decrease after the sudden increase, assuming a linear decrease for simplicity
        decrease_factor = 1 - (time_since_CS - CS_duration) / (US_time - CS_duration)
        return baseline * (1 + (0.5 * decrease_factor))
    elif time_since_CS >= US_time:
        # Large spike at US and then return to baseline
        return baseline * 3 if time_since_CS == US_time else baseline


def type_five_response(time_since_CS, baseline): #not moving, in field
    CS_duration = 0.25  # CS duration in seconds
    CS_to_US_delay = 0.5  # Delay between CS and US in seconds
    US_time = CS_duration + CS_to_US_delay  # Time when US occurs
    total_time = CS_duration + CS_to_US_delay + 0.25  # Total duration considered for response
    # Random fluctuation throughout the period
    fluctuation = np.random.uniform(-0.2, 0.2) * baseline
    return baseline + fluctuation
