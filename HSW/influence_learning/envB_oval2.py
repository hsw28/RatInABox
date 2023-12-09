import numpy as np
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from CombinedPlaceTebcNeurons2 import CombinedPlaceTebcNeurons
from trial_marker2 import determine_cs_us


#modeling environment B (oval)
#using equation from https://www.biorxiv.org/content/10.1101/2023.10.08.561112v1.full :
'''
Place and grid cell rate maps were generated from a real exploration trajectory using
the open source Python software RatInABox. The respective activity rates are then used
to train a logistic regressor to predict the real activity of each individual neurons.
To evaluate each model performance, we computed a F1 score for each neuron using
a place input model, which penalizes both incorrect classifications of active and inactive periods.
'''

#allows me to upload my own trajectory <-- I HAVE TO SCALE THIS
# Similar to EnvA, but with adjustments for EnvB dimensions and trajectory data


def simulate_envB(agent, position_data, balance_distribution_envA, tebc_responsive_rates_envA, tebc_responsive_neuronsA, cell_types_envA):


    N = 80

    # Define place cell parameters for EnvA
    place_cells_params_envB = {
        "n": N,  # Number of place cells
        "description": "gaussian",  # Adjust as needed for EnvA
        "widths": 0.20,  # Adjust as needed for EnvA
        "place_cell_centres": None,  # Adjust as needed for EnvA
        "wall_geometry": "geodesic",  # Adjust as needed for EnvA
        "min_fr": 0,  # Minimum firing rate
        "max_fr": 12,  # Maximum firing rate
        "save_history": True  # Save history for plotting
    }

    # Import trajectory into the agent
    times = position_data[0]
    positions = position_data[1:3].T
    unique_times, indices = np.unique(times, return_index=True)
    unique_positions = positions[indices]
    agent.import_trajectory(times=unique_times, positions=unique_positions)

    # Create CombinedPlaceTebcNeurons instance for EnvB
    combined_neurons = CombinedPlaceTebcNeurons(agent, N, balance_distribution_envA, tebc_responsive_rates_envA, place_cells_params_envB, tebc_responsive_neuronsA, cell_types_envA)
    combined_neurons.calculate_smoothed_velocity(position_data)
    firing_rates = np.zeros((N, position_data.shape[1]))



    # Initialize last CS and US times
    last_CS_time = None
    last_US_time = None


    # Simulation loop
    for index in range(unique_positions.shape[0]):
        # Current timestamp
        current_time = unique_times[index]

        # Update the agent
        agent.update()

        # Determine if CS or US is present
        trial_marker = position_data[3, index]
        cs_present, us_present = determine_cs_us(trial_marker)

        # Update last CS/US time if necessary
        if cs_present and (last_CS_time is None or times[index] > last_CS_time):
            last_CS_time = times[index]
        if us_present and (last_US_time is None or times[index] > last_US_time):
            last_US_time = times[index]

        # Calculate time since CS and US
        time_since_CS = times[index] - last_CS_time if last_CS_time is not None else -1
        time_since_US = times[index] - last_US_time if last_US_time is not None else -1

        # Retrieve the agent's current position from the history
        agent_position = agent.history['pos'][index]

        # Update neuron states
        combined_neurons.update_my_state(agent_position, time_since_CS, time_since_US, index)

        # Store firing rates
        firing_rates[:, index] = combined_neurons.get_firing_rates()


    # Return the firing rates for further analysis
    return firing_rates, agent, combined_neurons
