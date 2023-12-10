import numpy as np
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import Neurons, PlaceCells
from trial_marker2 import determine_cs_us
from TEBCcells import TEBC
import cProfile
import pstats

def simulate_envA(agent, position_data, balance_distribution, responsive_distribution, tebc_responsive_neurons, cell_types):
    N = 80  # Number of neurons

    # Define place cell parameters for EnvA
    PCs = PlaceCells(agent, params={
        "n": N,
        "description": "gaussian",
        "widths": 0.20,
        "place_cell_centres": None,
        "wall_geometry": "geodesic",
        "min_fr": 0,
        "max_fr": 12,
        "save_history": True
    })

    eyeblink_neurons = TEBC(agent, N, balance_distribution, responsive_distribution, PCs.params, tebc_responsive_neurons, cell_types)

    firing_rates = np.zeros((N, position_data.shape[1]))
    eyeblink_neurons.calculate_smoothed_velocity(position_data)

    last_CS_time = None
    last_US_time = None

    times = position_data[0, :]
    trial_markers = position_data[3, :]

    for index, (current_time, trial_marker) in enumerate(zip(times, trial_markers)):
        agent.update()

        cs_present, us_present = determine_cs_us(trial_marker)

        if cs_present:
            last_CS_time = current_time if last_CS_time is None else max(last_CS_time, current_time)
        if us_present:
            last_US_time = current_time if last_US_time is None else max(last_US_time, current_time)

        time_since_CS = current_time - last_CS_time if last_CS_time is not None else -1
        time_since_US = current_time - last_US_time if last_US_time is not None else -1

        PCs.update()

        place_firing = (1 - eyeblink_neurons.balance_distribution) * PCs.history['firingrate'][-1]
        tebc_firing = eyeblink_neurons.update_my_state(time_since_CS, index)

        firing_rates[:, index] = tebc_firing + place_firing

    return eyeblink_neurons, firing_rates, agent
