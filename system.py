import asyncio
import math
import os
from datetime import datetime
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

    async def initialize(self, sim_length: int, delta_t: float):
        self.sim_length = sim_length
        self.delta_t = delta_t

        self.total_stored = np.zeros(sim_length)
        self.total_capacity = np.zeros(sim_length)
        self.total_consumption = np.zeros(sim_length)
        self.total_generation = np.zeros(sim_length)
        self.total_available_generation = np.zeros(sim_length)

        components = self.generators + self.loads + self.storages
        await asyncio.gather(*(component.initialize(sim_length, delta_t) for component in components))

    async def simulate(self, sim_time: float, delta_t: float):
        """
        Simulate a system of components
        :param sim_time: Total simulation time in hours
        :param delta_t: Time between subsequent ticks in hours
        :return:
        """
        sim_length: int = math.ceil(sim_time / delta_t) + 1
        await self.initialize(sim_length, delta_t)
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

    def graph(self, figname=None, show_each=True):
        if figname is None:
            figname = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(f"figures/{figname}", exist_ok=True)

        sim_days = int(self.sim_length * self.delta_t / 24)

        def plt_plot(xs, y, *args, **kwargs):
            xs = [a[0] / 24 for a in np.array_split(xs, sim_days)]
            y = [np.average(a) for a in np.array_split(y, sim_days)]
            plt.plot(xs, y, *args, **kwargs)

        x = np.arange(self.sim_length) * self.delta_t

        # Plotting each graph on a different subplot
        plt_plot(x, self.total_stored, 'r', label="Total Stored")  # red line
        plt_plot(x, self.total_capacity, 'g', label="Total Capacity")  # green line
        plt.title("Energy Storage Over Time")
        plt.ylim(bottom=np.min(self.total_stored, initial=0))
        plt.xlabel("Time (days)")
        plt.ylabel("Energy (Wh)")
        plt.legend()
        plt.savefig(f"figures/{figname}/total_storage.png")
        plt.show()

        plt_plot(x, self.total_consumption, 'b')  # blue line
        plt.title("Total Consumption Over Time")
        plt.ylim(bottom=np.min(self.total_consumption, initial=0))
        plt.xlabel("Time (days)")
        plt.ylabel("Power (W)")
        plt.savefig(f"figures/{figname}/total_consumption.png")
        plt.show()

        plt_plot(x, self.total_generation, 'm', label="Total Generation")
        plt_plot(x, self.total_available_generation, 'r', label="Total Available")
        plt.title("Generation Over Time")
        plt.ylim(bottom=np.min(self.total_generation, initial=0))
        plt.xlabel("Time (days)")
        plt.ylabel("Power (W)")
        plt.legend()
        plt.savefig(f"figures/{figname}/total_generation.png")
        plt.show()

        if show_each:
            for i, generator in enumerate(self.generators):
                plt_plot(x, generator.generation_amounts, label=f"{generator.__class__.__name__} {i + 1}",
                         linestyle='--')
            plt.title("Generation Per Generator Over Time")
            plt.ylim(bottom=np.min(self.total_generation, initial=0))
            plt.xlabel("Time (days)")
            plt.ylabel("Power (W)")
            plt.legend()
            plt.savefig(f"figures/{figname}/generation_per_generator.png")
            plt.show()

    def stats(self, show_each=False):
        print(f"Total Generated: {np.sum(self.total_generation) * self.delta_t / 1000: .2f}kWh")
        print(f"Average Generated: {np.average(self.total_generation):.2f}W")

        if show_each:
            for i, gen in enumerate(self.generators):
                print(f"{gen.__class__.__name__} {i + 1}")
                print("\n".join([f"\t{k}: {v}" for k, v in gen.stats().items()]))

        print(f"Total Consumption: {np.sum(self.total_consumption) * self.delta_t / 1000:.2f}kWh")
        print(f"Average Consumption: {np.average(self.total_consumption):.2f}W")

    def get_initial_cost(self):
        return sum(component.get_initial_cost() for component in self.generators + self.storages)

    def get_operation_cost(self):
        return sum(component.get_operation_cost() for component in self.generators)
