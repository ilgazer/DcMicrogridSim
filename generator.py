import math
import random

import numpy as np

from utils import linear, limit


class Generator:
    generation_amounts: np.array
    availability: np.array
    sim_length: int
    delta_t: float

    def initialize(self, sim_length: int, delta_t: float):
        self.generation_amounts = np.zeros(sim_length)
        self.availability = np.zeros(sim_length)
        self.delta_t = delta_t
        self.sim_length = sim_length

    def generate(self, index: int) -> float:
        pass

    def policy(self, storage_ratio):
        pass


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
                self.availability[index + i] = i > outage_ind_len

    def generate(self, index: int) -> tuple[float, float]:
        available_power = self.availability[index] * self.capacity
        generated = self.generate_next_tick * available_power
        self.generation_amounts[index] = generated
        return generated, available_power

    def policy(self, storage_ratio):
        curve = linear(0.8, 1, 0.9, 0)
        self.generate_next_tick = limit(curve(storage_ratio), 0, 1)


class DieselGenerator(Generator):
    capacity: float
    generate_next_tick: float

    def __init__(self, capacity: float):
        self.capacity = capacity

    def initialize(self, sim_length: int, delta_t: float):
        super().initialize(sim_length, delta_t)
        self.generate_next_tick = True

    def generate(self, index: int) -> tuple[float, float]:
        generated = self.generate_next_tick * self.capacity
        self.generation_amounts[index] = generated
        return generated, self.capacity

    def policy(self, storage_ratio):
        if self.generate_next_tick:
            self.generate_next_tick = storage_ratio < 0.4
        else:
            self.generate_next_tick = storage_ratio < 0.2
