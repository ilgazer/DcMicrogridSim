import math
import random

import numpy as np

from utils import linear, limit


class Generator:
    generation_amounts: np.array
    available_amounts: np.array
    sim_length: int
    delta_t: float

    def initialize(self, sim_length: int, delta_t: float):
        self.generation_amounts = np.zeros(sim_length)
        self.available_amounts = np.zeros(sim_length)
        self.delta_t = delta_t
        self.sim_length = sim_length

    def generate(self, index: int) -> float:
        pass

    def policy(self, storage_ratio):
        pass

    def stats(self):
        return {
            "Available": np.sum(self.available_amounts) * self.delta_t / 1000,
            "Used": np.sum(self.generation_amounts) * self.delta_t / 1000,
            "Percent Utilization": np.sum(self.generation_amounts) / np.sum(self.available_amounts) * 100,
        }


class Grid(Generator):
    capacity: float
    outage_period: float
    outage_ind_period: int
    mean_outage_length: float
    outage_length_std: float
    generate_next_tick: float

    def __init__(self, capacity: float, outage_period: float, mean_outage_length: float, outage_length_std: float):
        self.capacity = capacity
        self.outage_period = outage_period
        self.mean_outage_length = mean_outage_length
        self.outage_length_std = outage_length_std

    def initialize(self, sim_length: int, delta_t: float):
        super().initialize(sim_length, delta_t)
        self.outage_ind_period = math.ceil(self.outage_period / self.delta_t)
        self.generate_next_tick = True

        for index in range(0, sim_length, self.outage_ind_period):
            outage_length = random.normalvariate(self.mean_outage_length, self.outage_length_std)
            outage_ind_len = math.floor(outage_length / self.delta_t)
            for i in range(min(self.outage_ind_period, self.sim_length - index)):
                self.available_amounts[index + i] = (i > outage_ind_len) * self.capacity

    def generate(self, index: int) -> tuple[float, float]:
        generated = self.generate_next_tick * self.available_amounts[index]
        self.generation_amounts[index] = generated
        return generated, self.available_amounts[index]

    def policy(self, storage_ratio):
        curve = linear(0.8, 1, 0.9, 0)
        self.generate_next_tick = limit(curve(storage_ratio), 0, 1)


class DieselGenerator(Generator):
    capacity: float
    generate_next_tick: float
    start_generator_ratio: float
    stop_generator_ratio: float

    def __init__(self, capacity: float, start_generator_ratio: float = 0.4, stop_generator_ratio: float = 0.2):
        self.capacity = capacity
        self.start_generator_ratio = start_generator_ratio
        self.stop_generator_ratio = stop_generator_ratio

    def initialize(self, sim_length: int, delta_t: float):
        super().initialize(sim_length, delta_t)
        self.available_amounts += self.capacity
        self.generate_next_tick = True

    def generate(self, index: int) -> tuple[float, float]:
        generated = self.generate_next_tick * self.capacity
        self.generation_amounts[index] = generated
        return generated, self.capacity

    def policy(self, storage_ratio):
        if self.generate_next_tick:
            self.generate_next_tick = storage_ratio < self.stop_generator_ratio
        else:
            self.generate_next_tick = storage_ratio < self.start_generator_ratio


