import math
import random
from functools import lru_cache

import numpy as np
import pandas as pd
from async_lru import alru_cache

from utils import linear, limit, generate_amounts_array


class Generator:
    fixed_cost: float
    cost_per_wh: float
    generation_amounts: np.array
    available_amounts: np.array
    sim_length: int
    delta_t: float

    def __init__(self, fixed_cost: float, cost_per_wh: float):
        self.fixed_cost = fixed_cost
        self.cost_per_wh = cost_per_wh

    async def initialize(self, sim_length: int, delta_t: float):
        self.generation_amounts = np.zeros(sim_length)
        self.available_amounts = np.zeros(sim_length)
        self.delta_t = delta_t
        self.sim_length = sim_length

    def generate(self, index: int) -> tuple[float, float]:
        return self.generation_amounts[index], self.available_amounts[index]

    def policy(self, storage_ratio):
        pass

    def stats(self):
        return {
            "Available": np.sum(self.available_amounts) * self.delta_t / 1000,
            "Used": np.sum(self.generation_amounts) * self.delta_t / 1000,
            "Percent Utilization": np.sum(self.generation_amounts) / np.sum(self.available_amounts) * 100,
        }

    def get_initial_cost(self):
        return self.fixed_cost

    def get_operation_cost(self):
        return self.cost_per_wh * self.delta_t * np.sum(self.generation_amounts)


class Grid(Generator):
    capacity: float
    outage_period: float
    outage_ind_period: int
    mean_outage_length: float
    outage_length_std: float
    generate_next_tick: float

    def __init__(self, capacity: float, outage_period: float, mean_outage_length: float, outage_length_std: float):
        # Cost from https://www.reuters.com/business/energy/ukraine-nearly-doubles-consumer-electricity-tariffs-help-power-sector-repairs-2024-05-31/
        super().__init__(0, 0.099 / 1000)
        self.capacity = capacity
        self.outage_period = outage_period
        self.mean_outage_length = mean_outage_length
        self.outage_length_std = outage_length_std

    async def initialize(self, sim_length: int, delta_t: float):
        await super().initialize(sim_length, delta_t)
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

    def __init__(self, price: float, cost_per_wh, capacity: float, start_generator_ratio: float = 0.4,
                 stop_generator_ratio: float = 0.2):
        super().__init__(price, cost_per_wh)
        self.capacity = capacity
        self.start_generator_ratio = start_generator_ratio
        self.stop_generator_ratio = stop_generator_ratio

    async def initialize(self, sim_length: int, delta_t: float):
        await super().initialize(sim_length, delta_t)
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


@lru_cache(maxsize=None)
def get_data():
    data = pd.read_csv('Systeem_Amstelveen/2022_15min_data.csv', sep=",")
    data["Timestamp"] = pd.to_datetime(data["Time"])
    data.set_index(["Timestamp"])
    data["Power Ratio"] = data["PV Productie (W)"] / 3080
    indices = data["Timestamp"].values.copy()
    print("Done with loading data")
    return data, indices


@alru_cache(maxsize=None)
async def generate_solar_amounts(sim_length: int, delta_t: float, start_timestamp: pd.Timestamp):
    data, indices = get_data()
    return await generate_amounts_array(indices, data["Power Ratio"], sim_length, delta_t, start_timestamp)


class SolarPanel(Generator):
    capacity: float
    start_timestamp: pd.Timestamp

    def __init__(self, capacity: float, start_timestamp: pd.Timestamp):
        """
        Initialise new solar panel backed by Systeem_Amstelveen data
        :param capacity: Capacity of the solar panel in Wp
        """
        # Cost of solar installation assumed to be the same as the current cost in poland. Data from:
        # https://www.statista.com/statistics/1124327/poland-average-unit-price-of-pv-installation-depending-on-power
        super().__init__(1.275 * capacity, 0)
        self.capacity = capacity
        self.start_timestamp = start_timestamp

    async def initialize(self, sim_length: int, delta_t: float):
        await super().initialize(sim_length, delta_t)
        ratios = await generate_solar_amounts(sim_length, delta_t, self.start_timestamp)
        self.available_amounts = self.generation_amounts = ratios * self.capacity
