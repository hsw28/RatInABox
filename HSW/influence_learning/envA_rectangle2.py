import numpy as np
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from CombinedPlaceTebcNeurons2 import CombinedPlaceTebcNeurons
from trial_marker2 import determine_cs_us

def simulate_envA(agent, position_data, balance_distribution, responsive_distribution, tebc_responsive_neurons, cell_types):
    # Number of neurons
    N = 80

    # Define place cell parameters for EnvA
    place_cells_params_envA = {
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
    times = position_data[0]  # Timestamps
    positions = position_data[1:3].T  # Positions (x, y)
    unique_times, indices = np.unique(times, return_index=True)
    unique_positions = positions[indices]
    agent.import_trajectory(times=unique_times, positions=unique_positions)

    # Create CombinedPlaceTebcNeurons instance for EnvA
    combined_neurons = CombinedPlaceTebcNeurons(agent, N, balance_distribution, responsive_distribution, place_cells_params_envA, tebc_responsive_neurons, cell_types)
    firing_rates = np.zeros((N, position_data.shape[1]))
    combined_neurons.calculate_smoothed_velocity(position_data)


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
        combined_neurons.update_state(agent_position, time_since_CS, time_since_US, index)

        # Store firing rates
        firing_rates[:, index] = combined_neurons.get_firing_rates()

    # Return the firing rates for further analysis


    return firing_rates, agent, combined_neurons
