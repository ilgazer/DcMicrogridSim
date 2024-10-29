import math
from typing import Union

import numpy as np
from matplotlib import pyplot as plt

from generator import Generator
from load import Load
from storage import Storage


class System:
    generators: list[Generator]
    loads: list[Load]
    storages: list[Storage]

    total_stored: np.array
    total_capacity: np.array
    total_consumption: np.array
    total_generation: np.array
    total_available_generation: np.array

    sim_length: int
    delta_t: float

    def __init__(self):
        self.generators = []
        self.loads = []
        self.storages = []

    def initialize(self, sim_length: int, delta_t: float):
        self.sim_length = sim_length
        self.delta_t = delta_t

        self.total_stored = np.zeros(sim_length)
        self.total_capacity = np.zeros(sim_length)
        self.total_consumption = np.zeros(sim_length)
        self.total_generation = np.zeros(sim_length)
        self.total_available_generation = np.zeros(sim_length)

        for component in self.generators + self.loads + self.storages:
            component.initialize(sim_length, delta_t)

    def simulate(self, time: float, delta_t: float):
        """
        Simulate a system of components
        :param time: Total simulation time in hours
        :param delta_t: Time between subsequent ticks in hours
        :return:
        """
        sim_length: int = math.ceil(time / delta_t) + 1
        self.initialize(sim_length, delta_t)
        for storage in self.storages:
            self.total_capacity[0] += storage.storage_capacities[0]
            self.total_stored[0] += storage.storage_amounts[0]

        for index in range(1, sim_length):
            total_generation = 0
            total_consumption = 0
            for load in self.loads:
                total_consumption += load.consume(index)
            self.total_consumption[index] = total_consumption

            for generator in self.generators:
                generated, available_power = generator.generate(index)
                total_generation += generated
                self.total_available_generation[index] += available_power

            self.total_generation[index] = total_generation

            power_ratio = (total_generation - total_consumption) / self.total_capacity[index - 1] * delta_t
            total_stored = 0
            total_capacity = 0
            for storage in self.storages:
                capacity, stored = storage.update(index, power_ratio)
                total_capacity += capacity
                total_stored += stored

            self.total_stored[index] = total_stored
            self.total_capacity[index] = total_capacity

            for generator in self.generators:
                generator.policy(total_stored / total_capacity)

    def add(self, *components: Union[Generator, Load, Storage]) -> "System":
        for component in components:
            if isinstance(component, Generator):
                self.generators.append(component)
            elif isinstance(component, Load):
                self.loads.append(component)
            elif isinstance(component, Storage):
                self.storages.append(component)
            else:
                raise ValueError("Unsupported component")

        return self

    def graph(self, show_each_gen=False):
        x = np.arange(self.sim_length) * self.delta_t
        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Plotting each graph on a different subplot
        axs[0, 0].plot(x, self.total_stored, 'r', label="Total Stored")  # red line
        axs[0, 0].plot(x, self.total_capacity, 'g', label="Total Capacity")  # green line
        axs[0, 0].set_title("Batteries Over Time")
        axs[0, 0].set_ylim(bottom=np.min(self.total_stored, initial=0))
        axs[0, 0].legend()

        axs[1, 0].plot(x, self.total_consumption, 'b')  # blue line
        axs[1, 0].set_title("Total Consumption Over Time")
        axs[1, 0].set_ylim(bottom=np.min(self.total_consumption, initial=0))

        axs[1, 1].plot(x, self.total_generation, 'm', label="Total Generation")
        axs[1, 1].plot(x, self.total_available_generation, 'r', label="Total Available")
        if show_each_gen:
            for i, generator in enumerate(self.generators):
                axs[1, 1].plot(x, generator.generation_amounts, label=f"{generator.__class__.__name__} {i + 1}",
                               linestyle='--')
        axs[1, 1].set_title("Generation Over Time")
        axs[1, 1].set_ylim(bottom=np.min(self.total_generation, initial=0))
        axs[1, 1].legend()

        # Set common labels
        fig.text(0.5, 0.04, 'x', ha='center')
        fig.text(0.04, 0.5, 'y', va='center', rotation='vertical')

        # Adjust layout for better fit
        plt.tight_layout()
        plt.show()
