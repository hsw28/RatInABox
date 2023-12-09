import numpy as np
import pandas as pd
from ratinabox.Neurons import Neurons, PlaceCells
from tebc_response2 import response_profiles
from TEBCcells import TEBC

'''
Python class template for CombinedPlaceTebcNeurons that integrates both place cell and tEBC
cell functionalities. This class is designed to be used with the RatInABox framework.
- It includes a balance parameter to adjust the contribution of place cell activity versus
tEBC cell activity for each neuron.
- also includes tebc_responsive_rate that specifies the percentage of neurons that are responsive to tEBC signals.

# Example usage
num_neurons = 100
balance = 0.5  # Example balance factor
tebc_responsive_rate = 0.6  # Example: 60% of neurons are tEBC-responsive
combined_neurons = CombinedPlaceTebcNeurons(num_neurons, place_cells, balance, tebc_responsive_rate)

'''


class VelocityPlaceCells(PlaceCells):
    default_params = dict()  # Add this line to define the default_params attribute
    def __init__(self, agent, N, balance_distribution, responsive_distribution, place_cells_params, tebc_responsive_neurons=None, cell_types=None):
        super().__init__(agent, place_cells_params)

        # Define parameters for PlaceCells
        place_cells_params = {
            "n": N,  # Number of place cells
            "description": "gaussian",  # Example parameter, adjust as needed
            "widths": 0.20,  # Adjust as needed
            "place_cell_centres": None,  # Adjust as needed
            "wall_geometry": "geodesic",  # Adjust as needed
            "min_fr": 0,  # Adjust as needed
            "max_fr": 12,  # Adjust as needed
            "save_history": True  # Save history for plotting -- dont think this done anything
        }

        # Initialize tebc_responsive_neurons with a default value if not provided
        if tebc_responsive_neurons is not None:
            self.tebc_responsive_neurons = tebc_responsive_neurons
        else:
            self.tebc_responsive_neurons = np.full(N, False)  # Default value: all False

        # Initialize additional properties for CombinedPlaceTebcNeurons

        if cell_types is not None:
            self.cell_types = cell_types
        else:
            self.cell_types = np.full(N, False)  # Default value: all False

        self.agent = agent
        self.num_neurons = N
        self.balance_distribution = balance_distribution
        self.responsive_distribution = responsive_distribution
        self.firing_rates = np.zeros(N)
        self.history = {'t': [], 'firingrate': [], 'spikes': []}


    def calculate_smoothed_velocity(self, position_data):
        times = position_data[0, :]   # Timestamps
        xpos = position_data[1, :]    # X positions
        ypos = position_data[2, :]    # Y positions

        vel_vector = []
        s = len(times)

        for i in range(1, s - 1):
            if times[i] != times[i - 1]:
                hypo = np.hypot(xpos[i - 1] - xpos[i + 1], ypos[i - 1] - ypos[i + 1])
                vel = hypo / (times[i + 1] - times[i - 1])
                vel_vector.append(vel)

        # Smooth the velocity data
        window_size = 15
        self.smoothed_velocity = pd.Series(vel_vector).rolling(window=window_size, min_periods=1, center=True).mean().tolist()


    def update_my_state(self, agent_position, current_index):
        # Check the current smoothed velocity
        current_velocity = self.smoothed_velocity[current_index] if current_index < len(self.smoothed_velocity) else 0
        #self.agent.position = agent_position
        self.update(save_history=False)

        fr = []

        print(len(self.history['firingrate']))
        for i in range(self.num_neurons):
            if current_velocity < 0.02:
                place_response = 0

            else:
                if i < len(self.history['firingrate']) and len(self.history['firingrate'][i]) > 0:
                    place_response = self.history['firingrate'][i][-1]
                    print(place_response)
                else:
                    place_response = 0  # Or some default value
                #    print("here2")


            self.firing_rates[i] = place_response * (1 - self.balance_distribution[i])


            self.history['firingrate'][i][-1] = self.firing_rates[i]


        return self.history['firingrate']


    def calculate_firing_rate(self, agent_position, time_since_CS, time_since_US):
        firing_rates = np.zeros(self.num_neurons)
        for i in range(self.num_neurons):
            place_response = self.firing_rates[i]  # Directly use the updated firing rates
            tebc_response = 0
            if self.tebc_responsive_neurons[i]:
                cell_type = self.cell_types[i]
                response_func = response_profiles[cell_type]['response_func']
                tebc_response = response_func(time_since_CS, time_since_US)
            firing_rates[i] = (1 - self.balance_distribution[i]) * place_response + self.balance_distribution[i] * tebc_response
        return firing_rates

    def update_tebc_response(self, retain_tebc_response):
        """
        Update or retain the tEBC response.
        retain_tebc_response: Boolean indicating whether to retain the existing tEBC response.
        """
        if not retain_tebc_response:
            # Reset or recalculate the tEBC response component
            self.tebc_responsive_neurons = self.assign_tebc_responsiveness_and_types()

    def get_firing_rates(self):
        # Return the current firing rates of all neurons
        return self.firing_rates

    def plot_rate_timeseries(self):
        # This method acts as a wrapper to the parent class's plot_ratemap method
        super(CombinedPlaceTebcNeurons, self).plot_rate_timeseries()

    def plot_rate_map(self):
        # This method acts as a wrapper to the parent class's plot_ratemap method
        super(CombinedPlaceTebcNeurons, self).plot_rate_map()

    def plot_place_cell_locations(self):
        # This method acts as a wrapper to the parent class's plot_ratemap method
        super(CombinedPlaceTebcNeurons, self).plot_place_cell_locations()
